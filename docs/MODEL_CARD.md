# Model card: directed-speech-detector (v0.1.0)

## Summary
A small PyTorch classifier that outputs the probability that an utterance is **directed at a voice assistant** (vs background speech). Variants: audio-only, text-only, audio + text, audio + text + enrolled-speaker.

**Important:** every result here is on a **synthetic proxy dataset** (formant-synthesised "speech", template transcripts). The model has never seen real speech or real assistant traffic. It is a portfolio/engineering artefact, **not** a production model and not a research claim about real-world accuracy.

## Intended use
- Demonstrating a reproducible multimodal pipeline: training, calibration at a target false-accept rate, ablations, INT8/ONNX export, serving.
- Starting point for experiments on real data via the manifest loader (`data/manifest.py`).

**Out of scope:** deciding whether to activate a real assistant, any security/privacy-critical gating, any claim about demographic groups or languages.

## Model details
| | |
|---|---|
| Inputs | 16 kHz audio (≤ 4 s) → 40-bin log-mel (25 ms / 10 ms); transcript tokens (word-level, ≤ 24); optional enrollment clip → mean/std log-mel profile |
| Audio branch | 2 strided conv layers (÷4 in time) + 3 pre-LN transformer blocks (d=96, 4 heads) + attention pooling |
| Text branch | embedding + 2 transformer blocks + attention pooling (vocabulary built from the training split) |
| Speaker branch | shared MLP over utterance/enrollment statistics, gated interaction (product, \|diff\|, cosine); learned null when no enrollment |
| Fusion | concatenation → GLU → linear logit; modality dropout 0.1 in training |
| Parameters | 552,771 (audio + text + speaker); 308,930 audio-only; 189,794 text-only |
| Training | AdamW, lr 2e-3 warmup + cosine, batch 64, ≤ 12 epochs, early stopping on validation AUC, word dropout 5 %, enrollment dropout 20 % |
| Calibration | temperature scaling + threshold for 5% FAR, both fit on the validation split |
| Exports | dynamic INT8 (linear layers), ONNX opset 17 (parity max \|Δlogit\| 3.8e-6) |

## Training and evaluation data
Synthetic proxy, household-disjoint splits: 6,000 train / 1,000 validation / 2,000 test utterances. See `DATA.md` for the generator, built-in assumptions and caveats.

## Measured performance (synthetic test split, 3 training seeds, 5% FAR operating point)
| Model | Accuracy % | FRR % | EER % |
|---|---:|---:|---:|
| Audio only | 82.5 ± 0.6 | 29.4 ± 1.4 | 13.6 ± 0.9 |
| Text only | 61.8 ± 0.8 | 70.7 ± 2.3 | 17.3 ± 0.2 |
| Audio + text | 90.8 ± 0.1 | 13.0 ± 0.4 | 7.5 ± 0.1 |
| Audio + text + speaker | 91.4 ± 0.4 | 11.3 ± 0.5 | 7.8 ± 0.4 |

Realised test FAR at the validation-derived threshold: 5.8 % for the speaker model (target 5 %). Efficiency (audio + text + speaker, one CPU thread): 2.24 MB / 0.88 MB (INT8) / 2.33 MB (ONNX); batch-1 p50 latency 2.57 / 2.92 / 1.74 ms (FP32 / INT8 / ONNX). Per-condition analysis: `results/error_analysis.md`.

## Known failure modes (measured on the proxy)
- Short utterances (1–2 s) are the hardest (accuracy 87.4 % vs 94.6 % for ≥ 3 s, seed 0).
- Heavy reverberation is treated as evidence of background speech (FRR 50 % at RT60 ≥ 0.5 s): a distance shortcut that the generator bakes in.
- The enrolled user's own background speech is a main error source (accuracy 90.3 % for the enrolled user vs 94.6 % for others).
- Text is ambiguous for sentences that are valid in both classes; text-only is poor at strict FAR.
- At a 1% FAR operating point the best model's FRR is still 48–59 % at this training scale.

## Limitations and risks
- No real-data validation; absolute numbers are meaningless outside the proxy.
- English-only template vocabulary (218 words); no evaluation of accents, dialects, ages, languages, speech impairments, or noise types beyond pink/synthetic babble. Speaker voices are random synthetic draws, so **no fairness conclusions can be drawn**; real deployment needs subgroup evaluation.
- Class prior 50/50; deployed prevalence is far lower, which lowers precision.
- An enrollment-conditioned model processes biometric-like voice information; real use needs consent, on-device storage of profiles and a privacy review. The API in this repo has no authentication or rate limiting.
- Calibration is fit on 1,000 validation utterances; realised FAR deviates from the target (4–6 %).

## Responsible-use notes
Report FAR/FRR together, choose operating points with product and privacy stakeholders, monitor drift per condition, and keep a human-reviewed path for contested false accepts. Re-evaluate on real, consented, demographically diverse data before any use beyond demonstration.
