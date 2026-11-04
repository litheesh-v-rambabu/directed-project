"""Checkpoint save/load (weights + config + vocabulary in one self-describing file)."""

from __future__ import annotations

from pathlib import Path

import torch

from directed_speech_detector.config import ExperimentConfig, config_from_dict
from directed_speech_detector.data.text import Vocabulary
from directed_speech_detector.models.detector import DirectedSpeechDetector


def build_model(cfg: ExperimentConfig, vocab_size: int) -> DirectedSpeechDetector:
    return DirectedSpeechDetector(cfg.model, vocab_size, cfg.features.n_mels)


def save_checkpoint(
    path: Path, model: DirectedSpeechDetector, cfg: ExperimentConfig, vocab: Vocabulary
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "vocab": vocab.to_list(), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path) -> tuple[DirectedSpeechDetector, ExperimentConfig, Vocabulary]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    cfg = config_from_dict(blob["config"])
    vocab = Vocabulary(blob["vocab"])
    model = build_model(cfg, len(vocab))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg, vocab
