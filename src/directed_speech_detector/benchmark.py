"""Benchmark FP32 vs dynamic-INT8 vs ONNX Runtime: latency, throughput, size and accuracy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from directed_speech_detector.config import FeatureConfig
from directed_speech_detector.data.dataset import DirectedSpeechDataset, collate_batch, get_split
from directed_speech_detector.evaluation import metrics as mx
from directed_speech_detector.evaluation.plots import plot_latency_size
from directed_speech_detector.export import (
    Backend,
    OnnxPredictor,
    export_all,
    quantize_int8,
    state_dict_size_mb,
)
from directed_speech_detector.features import LogMelExtractor
from directed_speech_detector.models.detector import INPUT_NAMES, DirectedSpeechDetector
from directed_speech_detector.training.checkpoint import load_checkpoint
from directed_speech_detector.utils import hardware_summary


def torch_backend(model: DirectedSpeechDetector) -> Backend:
    def run(batch: dict[str, torch.Tensor]) -> np.ndarray:
        with torch.inference_mode():
            return model(*(batch[n] for n in INPUT_NAMES)).numpy()

    return run


def time_calls(
    fn: Backend, batches: list[dict[str, torch.Tensor]], warmup: int, iterations: int
) -> np.ndarray:
    for i in range(warmup):
        fn(batches[i % len(batches)])
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        fn(batches[i % len(batches)])
        times.append((time.perf_counter() - start) * 1000.0)
    return np.array(times)


def feature_extraction_ms(cfg_features: FeatureConfig, seconds: float = 2.5, repeats: int = 100) -> float:
    extractor = LogMelExtractor(cfg_features)
    wave = torch.randn(1, int(seconds * 16_000)) * 0.05
    times = []
    for i in range(repeats + 10):
        start = time.perf_counter()
        extractor(wave)
        if i >= 10:
            times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))


def run_benchmark(run_dir: Path, docs_dir: Path, n_latency: int = 300, seed: int = 0) -> dict[str, Any]:
    """Benchmark all backends single-threaded on the test split and write docs/results/benchmark.*."""
    torch.set_num_threads(1)
    model, cfg, vocab = load_checkpoint(run_dir / "model.pt")
    export_all(run_dir)
    onnx_path = run_dir / "export" / "model.onnx"
    calibration = json.loads((run_dir / "calibration.json").read_text())

    test = get_split(cfg, "test")
    ds = DirectedSpeechDataset(test, vocab, cfg.model.max_text_len)
    items = [ds[i] for i in range(len(ds))]
    labels = test.labels
    singles = [collate_batch([items[i]]) for i in np.random.default_rng(seed).permutation(len(items))[:100]]
    bulk = [collate_batch([items[(i + j) % len(items)] for j in range(32)]) for i in range(0, 32 * 8, 32)]
    eval_batches = [collate_batch(items[i : i + 64]) for i in range(0, len(items), 64)]

    backends: dict[str, tuple[Backend, float]] = {
        "fp32": (torch_backend(model), state_dict_size_mb(model)),
        "int8": (torch_backend(quantize_int8(model)), state_dict_size_mb(quantize_int8(model))),
        "onnx": (OnnxPredictor(onnx_path), onnx_path.stat().st_size / 1e6),
    }
    rows = []
    for name, (fn, size_mb) in backends.items():
        logits = np.concatenate([fn(b) for b in eval_batches])
        probs = mx.sigmoid(logits / calibration["temperature"])
        rep = mx.classification_report(probs, labels, calibration["threshold"])
        single = time_calls(fn, singles, warmup=20, iterations=n_latency)
        bulk_ms = time_calls(fn, bulk, warmup=3, iterations=40)
        rows.append(
            {
                "backend": name,
                "size_mb": size_mb,
                "latency_p50_ms": float(np.percentile(single, 50)),
                "latency_p95_ms": float(np.percentile(single, 95)),
                "throughput_utt_per_s_batch32": float(32 / (np.median(bulk_ms) / 1000.0)),
                "accuracy_at_fp32_threshold": rep["accuracy"],
                "frr_at_fp32_threshold": rep["frr"],
                "far_at_fp32_threshold": rep["far"],
                "auc": mx.auc_score(logits, labels),
                "eer": mx.equal_error_rate(logits, labels),
                "max_abs_logit_diff_vs_fp32": float(
                    np.abs(logits - np.concatenate([backends["fp32"][0](b) for b in eval_batches])).max()
                ),
            }
        )
    result = {
        "run": cfg.name,
        "modalities": cfg.model.modalities,
        "n_parameters": model.num_parameters(),
        "threads": 1,
        "latency_iterations": n_latency,
        "n_test": len(items),
        "feature_extraction_p50_ms_2.5s_audio": feature_extraction_ms(cfg.features),
        "hardware": hardware_summary(),
        "rows": rows,
    }
    out = docs_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(result, indent=2))
    (out / "benchmark.md").write_text(benchmark_markdown(result))
    plot_latency_size(rows, docs_dir / "figures" / "latency_size.png")
    return result


def benchmark_markdown(result: dict[str, Any]) -> str:
    hw = result["hardware"]
    lines = [
        f"Model `{result['run']}` ({result['n_parameters']:,} parameters). {hw['cpu']}, 1 thread, "
        f"torch {hw['torch']}, {result['latency_iterations']} timed batch-1 calls on test utterances "
        "(model forward only; log-mel extraction for 2.5 s of audio takes "
        f"{result['feature_extraction_p50_ms_2.5s_audio']:.2f} ms p50). "
        "Accuracy and FRR/FAR use the FP32 model's calibrated 5%-FAR threshold for every backend.",
        "",
        "| Backend | Size MB | p50 ms | p95 ms | Throughput utt/s (batch 32) | Accuracy % | FRR % | FAR % | EER % | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["rows"]:
        lines.append(
            f"| {r['backend'].upper()} | {r['size_mb']:.2f} | {r['latency_p50_ms']:.2f} | {r['latency_p95_ms']:.2f} | "
            f"{r['throughput_utt_per_s_batch32']:.0f} | {r['accuracy_at_fp32_threshold'] * 100:.2f} | "
            f"{r['frr_at_fp32_threshold'] * 100:.2f} | {r['far_at_fp32_threshold'] * 100:.2f} | {r['eer'] * 100:.2f} | {r['auc']:.4f} |"
        )
    return "\n".join(lines) + "\n"
