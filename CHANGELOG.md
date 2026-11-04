# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and [SemVer](https://semver.org/).

## [0.1.0] - 2026-11-04
### Added
- Synthetic proxy data generator (formant audio, reverb/noise, ambiguous templates, ASR-style errors, household enrollment).
- Audio, text and enrolled-speaker branches with gated fusion and modality dropout; YAML-driven configs and `dsd` CLI.
- Evaluation: accuracy, precision/recall/F1, FAR/FRR, EER, temperature calibration, threshold at target FAR, per-condition analysis, plots.
- Dynamic INT8 quantization, ONNX export with parity check, FP32/INT8/ONNX benchmark.
- FastAPI service (`/predict`, `/health`, `/metrics`), Dockerfile and docker-compose.
- pytest suite, ruff, mypy, pre-commit, GitHub Actions CI, Makefile.
- Colab notebook and configs for a full-scale GPU run (not yet run); manifest-based real-audio data path.
