"""Matplotlib figures for docs/figures (headless backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from directed_speech_detector.evaluation.metrics import roc_curve

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
DISPLAY = {
    "audio_only": "Audio only",
    "text_only": "Text only",
    "fused": "Audio + text",
    "fused_speaker": "Audio + text + speaker",
}


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_tradeoff(preds: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    """False-reject rate vs false-accept rate (DET-style, log-x) for each model."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for color, (name, (logits, labels)) in zip(PALETTE, preds.items(), strict=False):
        _, far, frr = roc_curve(logits, labels.astype(int))
        ax.plot(np.maximum(far, 1e-3), frr, label=DISPLAY.get(name, name), color=color, lw=2)
    ax.axvline(0.05, color="grey", ls="--", lw=1)
    ax.text(0.052, 0.9, "5% FAR", color="grey", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("False-accept rate (background accepted)")
    ax.set_ylabel("False-reject rate (directed missed)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _finish(fig, path)


def plot_reliability(bins: dict[str, list[dict[str, float]]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    for color, (name, rows) in zip(PALETTE, bins.items(), strict=False):
        ax.plot(
            [r["confidence"] for r in rows],
            [r["accuracy"] for r in rows],
            marker="o",
            color=color,
            label=name,
        )
    ax.set_xlabel("Predicted P(directed)")
    ax.set_ylabel("Observed fraction directed")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _finish(fig, path)


def plot_conditions(
    per_model: dict[str, dict[str, list[dict[str, Any]]]], condition: str, path: Path, title: str
) -> None:
    """Grouped bars of accuracy by bucket of one condition variable, one bar per model."""
    names = list(per_model)
    buckets = [r["bucket"] for r in per_model[names[0]][condition]]
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, (color, name) in enumerate(zip(PALETTE, names, strict=False)):
        acc = [r["accuracy"] for r in per_model[name][condition]]
        ax.bar(np.arange(len(buckets)) + i * width, acc, width, label=DISPLAY.get(name, name), color=color)
    ax.set_xticks(np.arange(len(buckets)) + width * (len(names) - 1) / 2, buckets)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Accuracy @ 5% FAR threshold")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, path)


def plot_ablation(summary: dict[str, dict[str, tuple[float, float]]], path: Path) -> None:
    """Mean ± std (over seeds) of EER and FRR@5%FAR per ablation."""
    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, key, label in (
        (axes[0], "eer", "EER (lower is better)"),
        (axes[1], "frr_at_far5", "FRR @ 5% FAR (lower is better)"),
    ):
        means = [summary[n][key][0] for n in names]
        stds = [summary[n][key][1] for n in names]
        ax.bar(range(len(names)), means, yerr=stds, color=PALETTE[: len(names)], capsize=4)
        ax.set_xticks(
            range(len(names)), [DISPLAY.get(n, n).replace(" + ", "\n+ ") for n in names], fontsize=8
        )
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    _finish(fig, path)


def plot_latency_size(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    labels = [r["backend"] for r in rows]
    axes[0].bar(
        labels,
        [r["latency_p50_ms"] for r in rows],
        yerr=[[0] * len(rows), [r["latency_p95_ms"] - r["latency_p50_ms"] for r in rows]],
        color=PALETTE[: len(rows)],
        capsize=4,
    )
    axes[0].set_ylabel("Batch-1 latency (ms): p50, whisker = p95")
    axes[1].bar(labels, [r["size_mb"] for r in rows], color=PALETTE[: len(rows)])
    axes[1].set_ylabel("Model size on disk (MB)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    _finish(fig, path)
