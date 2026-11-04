"""Typed, YAML-driven experiment configuration."""

from __future__ import annotations

import copy
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

VALID_MODALITIES = ("audio", "text", "speaker")


@dataclass
class DataConfig:
    """Synthetic-data generation settings (see docs/DATA.md)."""

    n_train: int = 6000
    n_val: int = 1000
    n_test: int = 2000
    utterances_per_household: int = 25
    max_seconds: float = 4.0
    sample_rate: int = 16_000
    seed: int = 1234
    cache_dir: str = "data/cache"
    num_workers: int = 4
    manifest_dir: str | None = None  # directory with train.csv / val.csv / test.csv of real audio


@dataclass
class FeatureConfig:
    n_mels: int = 40
    n_fft: int = 400
    hop_length: int = 160
    f_min: float = 20.0


@dataclass
class ModelConfig:
    modalities: list[str] = field(default_factory=lambda: ["audio", "text", "speaker"])
    d_model: int = 96
    n_heads: int = 4
    ff_mult: int = 2
    audio_layers: int = 3
    text_layers: int = 2
    dropout: float = 0.1
    modality_dropout: float = 0.1
    max_text_len: int = 24


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-2
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    patience: int = 6
    seed: int = 0
    device: str = "auto"
    word_dropout: float = 0.05
    enrollment_dropout: float = 0.2


@dataclass
class EvalConfig:
    target_far: float = 0.05
    n_calibration_bins: int = 10
    snr_bins_db: list[float] = field(default_factory=lambda: [0.0, 8.0, 16.0, 24.0])
    duration_bins_s: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        bad = [m for m in self.model.modalities if m not in VALID_MODALITIES]
        if bad or not self.model.modalities:
            raise ValueError(f"modalities must be a non-empty subset of {VALID_MODALITIES}: {bad}")
        if self.model.d_model % self.model.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if self.data.manifest_dir and "speaker" in self.model.modalities:
            raise ValueError(
                "Manifest (real-audio) data has no enrollment profiles; drop the 'speaker' modality"
            )
        if not 0.0 < self.eval.target_far < 1.0:
            raise ValueError("eval.target_far must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively construct a dataclass from a plain dict, rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        hint = hints[key]
        nested = is_dataclass(hint) and isinstance(value, dict)
        kwargs[key] = _build(hint, value) if nested else value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    base_ref = raw.pop("base", None)
    if base_ref is None:
        return raw
    return _deep_merge(_load_yaml_with_base(path.parent / base_ref), raw)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (values parsed as YAML)."""
    out = copy.deepcopy(raw)
    for item in overrides:
        dotted, _, text = item.partition("=")
        if not text and "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        node = out
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(text)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a YAML config (supports a ``base:`` key for inheritance) and CLI overrides."""
    raw = apply_overrides(_load_yaml_with_base(Path(path)), overrides or [])
    cfg = _build(ExperimentConfig, raw)
    cfg.validate()
    return cfg


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    cfg = _build(ExperimentConfig, raw)
    cfg.validate()
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
