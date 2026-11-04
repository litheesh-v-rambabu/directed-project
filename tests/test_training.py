"""End-to-end smoke tests: train -> evaluate -> export -> serve on a tiny generated dataset."""

import json

import numpy as np
import pytest

from directed_speech_detector.cli import main
from directed_speech_detector.config import load_config
from directed_speech_detector.serving.inference import Detector
from directed_speech_detector.training.checkpoint import load_checkpoint
from directed_speech_detector.training.trainer import train_model

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    cfg = load_config(
        "configs/smoke.yaml",
        [
            f"output_dir={root / 'runs'}",
            f"data.cache_dir={root / 'cache'}",
            "train.epochs=3",
            "data.n_train=250",
            "data.n_val=100",
            "data.n_test=100",
        ],
    )
    result = train_model(cfg)
    return cfg, result


def test_training_produces_loadable_checkpoint(trained_run) -> None:
    cfg, result = trained_run
    model, loaded_cfg, vocab = load_checkpoint(result.run_dir / "model.pt")
    assert loaded_cfg.model.modalities == cfg.model.modalities and len(vocab) > 10
    history = json.loads((result.run_dir / "history.json").read_text())
    assert len(history["history"]) == 3 and result.best_val_auc > 0.6
    assert history["history"][-1]["train_loss"] < history["history"][0]["train_loss"]


def test_training_is_reproducible(trained_run, tmp_path) -> None:
    cfg, result = trained_run
    cfg2 = load_config(
        "configs/smoke.yaml",
        [
            f"output_dir={tmp_path}",
            f"data.cache_dir={cfg.data.cache_dir}",
            "train.epochs=3",
            "data.n_train=250",
            "data.n_val=100",
            "data.n_test=100",
        ],
    )
    again = train_model(cfg2)
    first, second = (json.loads((r.run_dir / "history.json").read_text())["history"] for r in (result, again))
    assert np.allclose([h["train_loss"] for h in first], [h["train_loss"] for h in second], atol=1e-5)


def test_evaluate_export_and_serve(trained_run) -> None:
    from directed_speech_detector.evaluation.analysis import evaluate_run
    from directed_speech_detector.export import export_all

    _, result = trained_run
    metrics = evaluate_run(result.run_dir)
    assert 0.0 <= metrics["eer"] <= 1.0 and "without_enrollment_at_inference" in metrics
    far_key = next(k for k in metrics["at_target_far"] if k.endswith("0.05"))
    assert metrics["at_target_far"][far_key]["far"] <= 0.15  # small val set => loose bound
    assert {"snr", "duration", "speaker"} <= set(metrics["per_condition"])
    report = export_all(result.run_dir)
    assert report["parity"]["passed"]
    audio = np.random.default_rng(0).normal(0, 0.05, 16_000).astype(np.float32)
    outputs = {
        b: Detector.from_run_dir(result.run_dir, b).predict(audio, "play some jazz").probability
        for b in ("fp32", "int8", "onnx")
    }
    assert abs(outputs["fp32"] - outputs["onnx"]) < 1e-3 and all(0 <= p <= 1 for p in outputs.values())


def test_cli_parser_and_overrides(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main([])
    assert (
        main(
            [
                "generate-data",
                "--config",
                "configs/smoke.yaml",
                "--set",
                f"data.cache_dir={tmp_path}",
                "data.n_train=20",
                "data.n_val=10",
                "data.n_test=10",
            ]
        )
        == 0
    )
