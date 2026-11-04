from pathlib import Path

import pytest

from directed_speech_detector.config import apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_base_inheritance_and_values() -> None:
    cfg = load_config(ROOT / "configs" / "audio_only.yaml")
    assert cfg.model.modalities == ["audio"]
    assert cfg.model.d_model == 96  # inherited from base.yaml
    assert cfg.name == "audio_only"


def test_overrides_are_typed_via_yaml() -> None:
    cfg = load_config(
        ROOT / "configs" / "fused.yaml", ["train.epochs=2", "train.lr=0.5", "model.modalities=[text]"]
    )
    assert cfg.train.epochs == 2 and cfg.train.lr == 0.5 and cfg.model.modalities == ["text"]


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  epochz: 3\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_invalid_modality_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  modalities: [video]\n")
    with pytest.raises(ValueError, match="modalities"):
        load_config(path)


def test_override_syntax_error() -> None:
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


@pytest.mark.parametrize("name", ["audio_only", "text_only", "fused", "fused_speaker", "smoke"])
def test_all_shipped_configs_load(name: str) -> None:
    assert load_config(ROOT / "configs" / f"{name}.yaml").name == name
