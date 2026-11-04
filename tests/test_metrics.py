import numpy as np
import pytest

from directed_speech_detector.evaluation import metrics as mx


def test_perfect_separation() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    assert mx.auc_score(s, y) == 1.0 and mx.equal_error_rate(s, y) == 0.0
    rep = mx.classification_report(mx.sigmoid(s), y, 0.5)
    assert rep["accuracy"] == 1.0 and rep["far"] == 0.0 and rep["frr"] == 0.0 and rep["f1"] == 1.0


def test_random_scores_give_chance_auc() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    assert abs(mx.auc_score(rng.normal(size=5000), y) - 0.5) < 0.03
    assert abs(mx.equal_error_rate(rng.normal(size=5000), y) - 0.5) < 0.04


def test_gaussian_eer_matches_theory() -> None:
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(20000), np.zeros(20000)].astype(int)
    s = np.r_[rng.normal(1, 1, 20000), rng.normal(-1, 1, 20000)]
    assert mx.equal_error_rate(s, y) == pytest.approx(0.1587, abs=0.01)  # Phi(-1)


def test_threshold_at_far_respects_target() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 4000)
    s = rng.normal(y * 1.5, 1.0)
    thr = mx.threshold_at_far(s, y, 0.05)
    assert mx.classification_report(s, y, thr)["far"] <= 0.05 + 1e-9
    assert (
        mx.classification_report(s, y, thr - 0.05)["far"] > mx.classification_report(s, y, thr)["far"] - 1e-12
    )


def test_report_counts_and_edge_cases() -> None:
    y = np.array([1, 1, 0, 0])
    rep = mx.classification_report(np.array([0.9, 0.2, 0.8, 0.1]), y, 0.5)
    assert (rep["tp"], rep["fn"], rep["fp"], rep["tn"]) == (1, 1, 1, 1)
    assert rep["precision"] == 0.5 and rep["recall"] == 0.5 and rep["far"] == 0.5 and rep["frr"] == 0.5
    with pytest.raises(ValueError):
        mx.threshold_at_far(np.array([0.1, 0.2]), np.array([1, 1]), 0.05)


def test_temperature_fit_recovers_overconfidence() -> None:
    rng = np.random.default_rng(3)
    true_logits = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < mx.sigmoid(true_logits)).astype(float)
    assert mx.fit_temperature(true_logits * 3.0, y) == pytest.approx(3.0, rel=0.1)


def test_ece_zero_for_calibrated_and_large_for_miscalibrated() -> None:
    rng = np.random.default_rng(4)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert mx.expected_calibration_error(p, y) < 0.02
    assert mx.expected_calibration_error(np.full(1000, 0.95), np.zeros(1000)) > 0.9


def test_grouped_report_buckets() -> None:
    rows = mx.grouped_report(
        np.array([0.9, 0.1, 0.9, 0.1]),
        np.array([1, 0, 0, 1]),
        0.5,
        np.array([1.0, 2.0, 11.0, 12.0]),
        [10.0],
        "dB",
    )
    assert [r["bucket"] for r in rows] == ["<10dB", ">=10dB"]
    assert rows[0]["accuracy"] == 1.0 and rows[1]["accuracy"] == 0.0
