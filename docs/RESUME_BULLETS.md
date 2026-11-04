# Resume bullets

**Project title:** Directed Speech Detector: On-Device Multimodal Device-Directed Speech Classification (PyTorch, ONNX, FastAPI)

All figures are measured in this repository on a **synthetic proxy dataset** (see `EXPERIMENTS.md`); mention the proxy in interviews.

1. Built an end-to-end PyTorch system that fuses a log-mel audio transformer, a transcript transformer and an enrolled-speaker branch via gated fusion with modality dropout to classify device-directed vs background speech, cutting equal-error rate from 13.6% (audio-only) to 7.5% (audio + text) on a synthetic proxy test set (3-seed means).
2. Designed a reproducible synthetic speech/transcript generator (formant synthesis, reverberation, babble noise, SNR-dependent ASR errors, household-disjoint enrollment) and ran 4 ablations × 3 seeds over 9,000 utterances, showing text-only reached 70.7% false-reject at a 5% false-accept target despite 0.904 AUC.
3. Calibrated the detector with validation-fit temperature scaling and a threshold at a 5% false-accept target, reaching 91.4% accuracy and 11.3% false-reject (5.8% realised FAR) with the audio + text + speaker model, and diagnosed condition-specific failures (87.4% accuracy on 1–2 s utterances vs 94.6% on ≥3 s; reverberation shortcut with 50% false-reject at RT60 ≥ 0.5 s).
4. Optimised for on-device inference with dynamic INT8 quantization (2.24 → 0.88 MB, 2.5× smaller, identical 91.8% test accuracy) and ONNX export validated by a parity test (max logit diff 3.8e-6), cutting single-thread batch-1 p50 latency from 2.57 ms to 1.74 ms (1.5×) on CPU, and reporting where INT8 did *not* help (2.92 ms p50).
5. Shipped production-style tooling: a FastAPI service with request validation, structured JSON logging and Prometheus-format latency metrics, Dockerfile/compose, a 67-test pytest suite (ONNX parity, padding invariance, API validation, end-to-end train→serve), mypy-strict/ruff, pre-commit and GitHub Actions CI; a test I wrote caught and fixed a batch-padding leak before any result was recorded.
