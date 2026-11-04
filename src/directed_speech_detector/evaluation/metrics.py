"""Pure-NumPy detection metrics. Convention: positive class = directed (label 1).

* False accept (FA): background speech accepted as directed. FAR = FA / #background.
* False reject (FR): directed speech rejected. FRR = FR / #directed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, far, frr) for every distinct threshold (accept if score >= thr)."""
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tp = np.cumsum(y)[idx]
    fp = (idx + 1) - tp
    n_pos, n_neg = float(y.sum()), float(y.size - y.sum())
    far = np.r_[0.0, fp / max(n_neg, 1.0)]
    frr = np.r_[1.0, 1.0 - tp / max(n_pos, 1.0)]
    thresholds = np.r_[np.inf, s[idx]]
    return thresholds, far, frr


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    _, far, frr = roc_curve(scores, labels)
    tpr = 1.0 - frr
    return float(np.trapezoid(tpr, far))


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER: the operating point where FAR == FRR (linear interpolation between thresholds)."""
    _, far, frr = roc_curve(scores, labels)
    diff = far - frr
    k = int(np.argmax(diff >= 0))
    if k == 0:
        return float(far[0])
    x0, x1 = diff[k - 1], diff[k]
    w = -x0 / (x1 - x0) if x1 != x0 else 0.0
    far_at = far[k - 1] + w * (far[k] - far[k - 1])
    frr_at = frr[k - 1] + w * (frr[k] - frr[k - 1])
    return float((far_at + frr_at) / 2.0)


def threshold_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> float:
    """Smallest threshold whose FAR on this (calibration) set is <= ``target_far``."""
    negatives = np.sort(scores[labels == 0])
    if negatives.size == 0:
        raise ValueError("Need background (label 0) examples to calibrate a FAR threshold")
    allowed = int(np.floor(target_far * negatives.size))
    # accept if score > thr: at most ``allowed`` negatives may exceed it
    return float(negatives[negatives.size - allowed - 1])


def classification_report(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probs > threshold).astype(int)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if tp > 0 else 0.0
    return {
        "threshold": float(threshold),
        "n": float(labels.size),
        "accuracy": _safe_div(tp + tn, labels.size),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": _safe_div(fp, fp + tn),
        "frr": _safe_div(fn, fn + tp),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if sel.any():
            ece += sel.mean() * abs(probs[sel].mean() - labels[sel].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if sel.any():
            out.append(
                {
                    "confidence": float(probs[sel].mean()),
                    "accuracy": float(labels[sel].mean()),
                    "count": float(sel.sum()),
                }
            )
    return out


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    z = logits / temperature
    return float(np.mean(np.logaddexp(0.0, z) - labels * z))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling: minimise NLL over T in [e^-2, e^3] via golden-section search."""
    lo, hi = -2.0, 3.0
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if negative_log_likelihood(logits, labels, np.exp(a)) < negative_log_likelihood(
            logits, labels, np.exp(b)
        ):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2.0))


def grouped_report(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    values: np.ndarray,
    edges: Sequence[float],
    unit: str,
) -> list[dict[str, Any]]:
    """Error rates per bucket of a continuous condition variable (e.g. SNR or duration)."""
    bounds = [-np.inf, *edges, np.inf]
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        sel = (values >= lo) & (values < hi)
        if not sel.any():
            continue
        name = (
            f"<{hi:g}{unit}"
            if np.isinf(lo)
            else f">={lo:g}{unit}"
            if np.isinf(hi)
            else f"{lo:g}-{hi:g}{unit}"
        )
        rep = classification_report(probs[sel], labels[sel], threshold)
        rows.append({"bucket": name, **{k: rep[k] for k in ("n", "accuracy", "far", "frr")}})
    return rows
