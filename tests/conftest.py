from __future__ import annotations

from pathlib import Path

import pytest
import torch

from directed_speech_detector.config import ExperimentConfig, load_config
from directed_speech_detector.data.dataset import (
    DirectedSpeechDataset,
    SplitArrays,
    collate_batch,
    fit_feature_stats,
    load_split,
)
from directed_speech_detector.data.text import Vocabulary
from directed_speech_detector.models.detector import DirectedSpeechDetector
from directed_speech_detector.training.checkpoint import build_model

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def mini_split() -> SplitArrays:
    return load_split(FIXTURES / "mini_split.npz")


@pytest.fixture(scope="session")
def vocab(mini_split: SplitArrays) -> Vocabulary:
    return Vocabulary.build(mini_split.texts)


@pytest.fixture()
def tiny_cfg() -> ExperimentConfig:
    return load_config(ROOT / "configs" / "smoke.yaml")


@pytest.fixture()
def mini_batch(
    mini_split: SplitArrays, vocab: Vocabulary, tiny_cfg: ExperimentConfig
) -> dict[str, torch.Tensor]:
    ds = DirectedSpeechDataset(mini_split, vocab, tiny_cfg.model.max_text_len)
    return collate_batch([ds[i] for i in range(8)])


@pytest.fixture()
def random_model(
    tiny_cfg: ExperimentConfig, vocab: Vocabulary, mini_split: SplitArrays
) -> DirectedSpeechDetector:
    torch.manual_seed(0)
    model = build_model(tiny_cfg, len(vocab))
    model.set_feature_stats(fit_feature_stats(mini_split))
    return model.eval()
