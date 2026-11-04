import numpy as np
import torch

from directed_speech_detector.data.dataset import (
    DirectedSpeechDataset,
    collate_batch,
    fit_feature_stats,
    load_split,
    save_split,
)


def test_fixture_contents(mini_split) -> None:
    assert len(mini_split) == 32
    assert mini_split.feats.shape[1] == 40 and mini_split.offsets[-1] == len(mini_split.feats)
    assert set(mini_split.labels.tolist()) <= {0, 1}
    assert {"snr_db", "duration_s", "is_enrolled_speaker", "household"} <= set(mini_split.meta)


def test_save_load_roundtrip(tmp_path, mini_split) -> None:
    save_split(tmp_path / "s.npz", mini_split)
    again = load_split(tmp_path / "s.npz")
    assert again.texts == mini_split.texts and np.array_equal(again.feats, mini_split.feats)


def test_collate_pads_and_masks(mini_split, vocab) -> None:
    ds = DirectedSpeechDataset(mini_split, vocab, 24)
    batch = collate_batch([ds[i] for i in range(4)])
    lengths = [ds[i]["feats"].shape[0] for i in range(4)]
    assert batch["feats"].shape == (4, max(lengths), 40)
    assert batch["feat_mask"].sum(dim=1).tolist() == [float(n) for n in lengths]
    assert batch["tokens"].dtype == torch.long and (batch["tokens"][:, 0] > 0).all()
    assert batch["enroll_profile"].shape == (4, 80)


def test_enrollment_dropout_and_word_dropout(mini_split, vocab) -> None:
    always_drop = DirectedSpeechDataset(mini_split, vocab, 24, enrollment_dropout=1.0)
    assert all(always_drop[i]["has_enroll"] == 0.0 for i in range(5))
    heavy = DirectedSpeechDataset(mini_split, vocab, 24, word_dropout=0.99)
    assert all(heavy[i]["tokens"].numel() >= 1 for i in range(len(heavy)))


def test_feature_stats_shapes(mini_split) -> None:
    stats = fit_feature_stats(mini_split)
    assert stats["mel_mean"].shape == (40,) and stats["profile_std"].shape == (80,)
    assert (stats["mel_std"] > 0).all()
