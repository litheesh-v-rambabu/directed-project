# Experiment log

Every number quoted in `README.md`, `MODEL_CARD.md` and `RESUME_BULLETS.md` comes from the runs below. All data is the **synthetic proxy** described in `DATA.md`.

**Environment (all runs):** Intel Xeon @ 2.10 GHz, 4 logical cores, no GPU, 15 GB RAM; Python 3.13.18, torch 2.15.0, onnxruntime 1.29.0. Raw artifacts (`runs/`, `data/`) are git-ignored; the aggregated outputs are committed under `docs/results/` and `docs/figures/`.

## E1. Ablation study (4 models × 3 seeds)
Configs: `configs/{audio_only,text_only,fused,fused_speaker}.yaml` (all inherit `configs/base.yaml`: d_model 96, 4 heads, 3 audio / 2 text layers, 12 epochs max, early stopping patience 4, AdamW lr 2e-3, batch 64, data seed 1234, 6,000/1,000/2,000 train/val/test utterances).

Commands (three processes in parallel, one per seed, single-threaded each):
```bash
for s in 0 1 2; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 dsd ablate --seeds $s & done; wait
dsd report            # writes docs/results/{ablation.md,ablation.json,error_analysis.md} and docs/figures/*.png
```
Each run was evaluated on the test split with the temperature and the 5%-FAR threshold fit on the validation split (`DECISIONS.md` D7). Results: `docs/results/ablation.md` (mean ± population std over seeds; std reflects training-seed variance only).

| Model | Seeds | Acc % @5% FAR (s0 / s1 / s2) | FRR % @5% FAR | Test FAR % @5% threshold | EER % | Train time s (seed 0, 1 thread, 3 jobs in parallel) | Best epoch (s0) |
|---|---|---|---|---|---|---:|---:|
| Audio only | 0,1,2 | 81.9 / 83.2 / 82.4 | 30.6 / 27.4 / 30.2 | 4.9 / 5.6 / 4.3 | see ablation.md | 466 | 7 |
| Text only | 0,1,2 | 62.9 / 61.3 / 61.1 | 67.5 / 71.9 / 72.7 | 4.7 / 3.6 / 3.1 | see ablation.md | 33 | 8 |
| Audio + text | 0,1,2 | 90.9 / 90.9 / 90.6 | 12.7 / 12.7 / 13.5 | 5.3 / 5.3 / 4.9 | see ablation.md | 540 | 8 |
| Audio + text + speaker | 0,1,2 | 91.8 / 91.4 / 90.9 | 10.6 / 11.6 / 11.7 | 5.7 / 5.4 / 6.4 | see ablation.md | 566 | 8 |

Operating point at **1% FAR** (same thresholds procedure) is much harder at this data scale: Audio + text + speaker reaches 71.2 / 69.4 / 75.0 % accuracy with FRR 55.8 / 58.9 / 47.6 % (seeds 0/1/2, test FAR 0.3–0.9 %). With the default 0.5 threshold on uncalibrated probabilities, the same model scores 92.9 / 91.8 / 92.4 % accuracy. Per-run details are in each run's `metrics.json` (not committed); the aggregates are committed.

Note on a discarded launch: the first ablation launch used a model that leaked padding through the conv stem; it was stopped after 5 epochs, deleted and re-run from scratch after the fix (`DECISIONS.md` D10).

## E2. Per-condition error analysis and confusion examples
Produced by `dsd report` from `runs/fused_speaker_s0` (single seed, test split, 5% FAR threshold) → `docs/results/error_analysis.md`, `docs/figures/accuracy_by_snr.png`, `accuracy_by_duration.png`, `error_tradeoff.png`, `reliability.png`.

## E3. Export parity and benchmark
```bash
dsd export    --run-dir runs/fused_speaker_s0     # writes runs/fused_speaker_s0/export/{model.onnx,export_report.json}
dsd benchmark --run-dir runs/fused_speaker_s0     # default --iterations 300; writes docs/results/benchmark.{md,json}, docs/figures/latency_size.png
```
- ONNX parity (opset 17, TorchScript exporter): max |Δlogit| = 3.8e-6 over batch sizes 16, 1 and 5 (tolerance 1e-3).
- Benchmark protocol: 1 intra-op thread; 300 timed batch-1 forwards cycling through 100 random test utterances (20 warm-up); throughput = batch of 32, median of 40 timed calls; accuracy uses the FP32 model's calibrated threshold for every backend. Latency covers the model forward only; log-mel extraction for 2.5 s of audio is 0.57 ms p50 on the same CPU.
- Single run on a shared sandbox VM: treat latency differences of a few tenths of a millisecond as noise-prone (only one benchmark run was made).

## E4. Live API smoke check
`dsd serve --run-dir runs/fused_speaker_s0 --backend onnx --port 8011` plus 12 `/predict` calls with synthetic test audio (11/12 decisions matched labels; one-off sanity check, not an evaluation), a malformed request (HTTP 422) and `/metrics`. Request total latency for the sample call was 4.91 ms (feature 2.52 ms, model 1.68 ms).

## E5. Tests / quality gates
`pytest` (67 tests, including slow end-to-end tests), `ruff check`, `ruff format --check`, `mypy` (strict) all pass locally on Python 3.13; final counts are in the README "Quality" section.

## Not yet run
| Item | Why |
|---|---|
| Full-scale GPU run (`configs/colab_full.yaml`, `notebooks/colab_full_training.ipynb`) | No GPU in the build environment |
| Real-audio manifest run (`configs/real_audio_template.yaml`) and any public dataset (SLURP, LibriSpeech, AMI, Common Voice) | Dataset hosts unreachable from the sandbox |
| Pretrained-encoder fine-tuning | Not implemented (`DECISIONS.md` D14) |
| `docker build` / `docker compose up` | Docker CLI present but no daemon in the sandbox; `docker compose config` validates the compose file only |
| Core ML export | Not attempted (`DECISIONS.md` D12) |
| GitHub Actions CI on Python 3.11/3.12 | Workflow committed; results are reported in the README only after a run is observed |
| Multi-seed benchmark / on-device (phone, NPU) latency | Single CPU benchmark only |
