# Decision log

Each entry records an autonomous decision, the reason, and the trade-off. Newest decisions are at the bottom.

## D1. Data/model path: from-scratch models on a synthetic proxy (environment-driven)
**Check run first.** PyPI was reachable (HTTP 200) but Hugging Face, OpenSLR, Zenodo returned no connection (`curl` status 000); no GPU (`nvidia-smi` absent), 4 logical CPUs, 15 GB RAM.
**Decision.** No pretrained checkpoints or public corpora could be downloaded, so the project uses compact from-scratch encoders trained on a documented synthetic generator. A Colab notebook and configs are provided for the full-scale run.
**Trade-off.** Absolute numbers say nothing about real assistant traffic; the value is the engineering, the controlled ablations and the reproducible pipeline.

## D2. Synthetic generator design (`data/synthetic.py`)
Formant-synthesised, syllable-timed voices with per-speaker pitch/vocal-tract parameters, exponential-decay reverberation, pink or babble noise at a sampled SNR, and slot-filled templates for transcripts with SNR-dependent ASR-style errors.
Directed speech is *on average* nearer-field, slower, with wider pitch range, and mostly spoken by the household's enrolled user; every distribution overlaps with background speech. These acoustic/speaker priors are assumptions I chose, not measurements of real devices (see `DATA.md`).

## D3. Text channel made deliberately ambiguous after a first trial
The first generator draft let a 4-epoch text-only model reach val AUC 0.99, which makes the multimodal question uninteresting (text alone solves the proxy). I changed the generator once so that half of the command-like background utterances reuse the exact directed templates and wake words are rarer (15% / 8% / 3%). I did not tune further after seeing ablation results. The first-trial numbers are not reported anywhere else because they came from a superseded generator.

## D4. Own transformer blocks built from `nn.Linear`
`nn.MultiheadAttention` complicates dynamic INT8 quantization and ONNX export. Plain linears + `scaled_dot_product_attention` with an additive mask make both exports predictable.

## D5. Features outside the model, normalisation inside
Log-mel extraction (STFT) stays in `features.py` (also used by the API). Per-dimension normalisers are model buffers fit on the training split only, so a checkpoint is self-contained and ONNX graphs need no STFT.

## D6. Speaker branch uses pooled log-mel statistics, not a pretrained speaker encoder
Input is the masked mean/std of log-mel frames for the utterance and for an enrollment clip; a shared MLP embeds both and a gated interaction (product, |diff|, cosine) is fused. With no enrollment, a learned null embedding is used (enrollment dropout 20% in training), so one model serves enrolled and un-enrolled users. It is a weak voice descriptor compared with an x-vector/ECAPA model; this is a known limitation.

## D7. Calibration and operating point
Temperature scaling is fit on the validation split; the decision threshold is the smallest validation threshold with FAR ≤ 5% (also 1% reported). Test metrics apply these val-derived values, so no test information is used to choose them. Positive class = directed. FAR = background accepted / all background; FRR = directed rejected / all directed. The generator's class prior is 50/50; real assistants see far fewer directed turns, so precision numbers do not transfer.

## D8. Ablation protocol
Four configurations (audio, text, audio+text, audio+text+speaker), 3 training seeds each (0, 1, 2), identical fixed data (data seed 1234). Seeds change initialisation, batch order and dropout, not the data, so reported ± is **training-seed variance only** (population std, `ddof=0`), not data-sampling variance. Early stopping and checkpoint selection use validation AUC.

## D9. ONNX via the TorchScript exporter (`dynamo=False`, opset 17)
Chosen for stable support of the ops used; dynamic batch/time axes. Parity test requires max |Δlogit| < 1e-3 across three batch shapes. Unused inputs would be pruned by ONNX, so the exported graph declares only the inputs a configuration consumes (`required_inputs`).

## D10. Bugs found by my own tests (and fixed before any reported run)
1. `torch.onnx.export` leaves the model in train mode (parity was comparing against dropout-active outputs). Fixed by restoring `eval()`.
2. Padded frames leaked into valid frames through the conv stem (batch composition changed predictions). Fixed by zeroing padded frames before each convolution; covered by `test_padding_does_not_change_output`. The first ablation launch used the pre-fix model and was **discarded and re-run**.
3. Tracing bug: tuple-unpacking a tensor in attention baked in shapes; replaced by `unbind`.

## D11. Dynamic INT8 quantizes `nn.Linear` only
Convolutions, embeddings and layer norms stay FP32 (PyTorch dynamic quantization limitation). Size and latency effects are reported as measured, whatever they are.

## D12. Core ML export not attempted
`coremltools` conversion cannot be validated for prediction parity on this Linux sandbox, so per the brief it is left out (see roadmap).

## D13. Handwritten Prometheus-format `/metrics`
A ~60-line registry (counters, histogram, rolling p50/p95) avoids an extra dependency in the slim image.

## D14. Pretrained-encoder fine-tuning not implemented
It could not be exercised offline; shipping untested code was rejected. Instead a manifest-based real-audio path (`data/manifest.py`, tested with generated WAVs) lets real corpora flow through the same from-scratch models. Fine-tuning wav2vec2/DistilBERT-class encoders is on the roadmap.

## D15. Tooling
`fork` start method for data-generation workers (workers are NumPy-only; `spawn` breaks when launched from stdin/notebooks). Ruff line length 110, mypy `strict` with `warn_return_any` off (torch's `Module.__call__` returns `Any`). CI targets Python 3.11/3.12; this sandbox verified Python 3.13 only.
