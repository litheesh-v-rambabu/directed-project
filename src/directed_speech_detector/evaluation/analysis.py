"""Full evaluation of a trained run: metrics, calibration, per-condition errors, examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from directed_speech_detector.config import ExperimentConfig
from directed_speech_detector.data.dataset import (
    DirectedSpeechDataset,
    SplitArrays,
    get_split,
    make_loader,
)
from directed_speech_detector.data.text import Vocabulary
from directed_speech_detector.evaluation import metrics as mx
from directed_speech_detector.models.detector import DirectedSpeechDetector
from directed_speech_detector.training.checkpoint import load_checkpoint
from directed_speech_detector.training.trainer import predict_logits

RT60_EDGES = (0.25, 0.5)


def split_logits(
    model: DirectedSpeechDetector,
    cfg: ExperimentConfig,
    vocab: Vocabulary,
    split: SplitArrays,
    force_no_enrollment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ds = DirectedSpeechDataset(split, vocab, cfg.model.max_text_len)
    loader = make_loader(ds, cfg.train.batch_size * 2, shuffle=False)
    return predict_logits(model, loader, torch.device("cpu"), force_no_enrollment)


def operating_point(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    rep = mx.classification_report(probs, labels, threshold)
    return {k: rep[k] for k in ("threshold", "accuracy", "precision", "recall", "f1", "far", "frr")}


def condition_report(
    probs: np.ndarray, labels: np.ndarray, threshold: float, split: SplitArrays, cfg: ExperimentConfig
) -> dict[str, list[dict[str, Any]]]:
    meta = split.meta
    enrolled = meta["is_enrolled_speaker"]
    by_speaker = []
    for value, name in ((1.0, "enrolled user"), (0.0, "other speaker")):
        sel = enrolled == value
        if sel.any():
            rep = mx.classification_report(probs[sel], labels[sel], threshold)
            by_speaker.append({"bucket": name, **{k: rep[k] for k in ("n", "accuracy", "far", "frr")}})
    return {
        "snr": mx.grouped_report(probs, labels, threshold, meta["snr_db"], cfg.eval.snr_bins_db, "dB"),
        "duration": mx.grouped_report(
            probs, labels, threshold, meta["duration_s"], cfg.eval.duration_bins_s, "s"
        ),
        "reverb_rt60": mx.grouped_report(probs, labels, threshold, meta["rt60"], RT60_EDGES, "s"),
        "speaker": by_speaker,
    }


def confusion_examples(
    probs: np.ndarray, labels: np.ndarray, threshold: float, split: SplitArrays, k: int = 8
) -> dict[str, list[dict[str, Any]]]:
    def rows(idx: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "text": split.texts[i],
                "p_directed": round(float(probs[i]), 3),
                "snr_db": round(float(split.meta["snr_db"][i]), 1),
                "enrolled_speaker": bool(split.meta["is_enrolled_speaker"][i]),
            }
            for i in idx
        ]

    fa = np.where((probs > threshold) & (labels == 0))[0]
    fr = np.where((probs <= threshold) & (labels == 1))[0]
    return {
        "false_accepts": rows(fa[np.argsort(-probs[fa])][:k]),
        "false_rejects": rows(fr[np.argsort(probs[fr])][:k]),
    }


def evaluate_run(run_dir: str | Path, write: bool = True) -> dict[str, Any]:
    """Evaluate ``run_dir/model.pt``: temperature + FAR thresholds from val, reported on test."""
    run_dir = Path(run_dir)
    model, cfg, vocab = load_checkpoint(run_dir / "model.pt")
    val, test = get_split(cfg, "val"), get_split(cfg, "test")
    val_logits, val_y = split_logits(model, cfg, vocab, val)
    test_logits, test_y = split_logits(model, cfg, vocab, test)
    val_y, test_y = val_y.astype(int), test_y.astype(int)

    temperature = mx.fit_temperature(val_logits, val_y.astype(float))
    val_cal, test_cal = mx.sigmoid(val_logits / temperature), mx.sigmoid(test_logits / temperature)
    fars = sorted({cfg.eval.target_far, 0.01}, reverse=True)
    thresholds = {f"far_{f:g}": mx.threshold_at_far(val_cal, val_y, f) for f in fars}
    primary = thresholds[f"far_{cfg.eval.target_far:g}"]

    result: dict[str, Any] = {
        "run": cfg.name,
        "modalities": cfg.model.modalities,
        "seed": cfg.train.seed,
        "n_parameters": model.num_parameters(),
        "n_test": int(test_y.size),
        "auc": mx.auc_score(test_logits, test_y),
        "eer": mx.equal_error_rate(test_logits, test_y),
        "temperature": temperature,
        "calibration": {
            "ece_uncalibrated": mx.expected_calibration_error(
                mx.sigmoid(test_logits), test_y, cfg.eval.n_calibration_bins
            ),
            "ece_calibrated": mx.expected_calibration_error(test_cal, test_y, cfg.eval.n_calibration_bins),
            "nll_uncalibrated": mx.negative_log_likelihood(test_logits, test_y),
            "nll_calibrated": mx.negative_log_likelihood(test_logits, test_y, temperature),
        },
        "at_default_threshold_0.5": operating_point(mx.sigmoid(test_logits), test_y, 0.5),
        "val_thresholds": thresholds,
        "at_target_far": {name: operating_point(test_cal, test_y, thr) for name, thr in thresholds.items()},
        "per_condition": condition_report(test_cal, test_y, primary, test, cfg),
        "confusion_examples": confusion_examples(test_cal, test_y, primary, test),
        "reliability_bins": {
            "uncalibrated": mx.reliability_bins(mx.sigmoid(test_logits), test_y),
            "calibrated": mx.reliability_bins(test_cal, test_y),
        },
    }
    if "speaker" in cfg.model.modalities:
        no_enroll, _ = split_logits(model, cfg, vocab, test, force_no_enrollment=True)
        result["without_enrollment_at_inference"] = {
            "auc": mx.auc_score(no_enroll, test_y),
            "eer": mx.equal_error_rate(no_enroll, test_y),
            **operating_point(mx.sigmoid(no_enroll / temperature), test_y, primary),
        }
    if write:
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        (run_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "temperature": temperature,
                    "threshold": primary,
                    "target_far": cfg.eval.target_far,
                    "thresholds": thresholds,
                },
                indent=2,
            )
        )
        np.savez(
            run_dir / "predictions.npz",
            logits=test_logits,
            labels=test_y,
            snr_db=test.meta["snr_db"],
            duration_s=test.meta["duration_s"],
        )
    return result
