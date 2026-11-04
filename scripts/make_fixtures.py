"""Regenerate the tiny offline test fixture (tests/fixtures/mini_split.npz)."""

from __future__ import annotations

from pathlib import Path

from directed_speech_detector.config import DataConfig, FeatureConfig
from directed_speech_detector.data.dataset import build_split, save_split

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mini_split.npz"

if __name__ == "__main__":
    data = DataConfig(n_train=0, n_val=0, n_test=0, utterances_per_household=8, seed=7, num_workers=1)
    split = build_split(data, FeatureConfig(), "test", 32)
    save_split(OUT, split)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(split)} utterances)")
