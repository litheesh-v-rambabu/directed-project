"""Cached feature datasets, batching and collation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from directed_speech_detector.config import DataConfig, ExperimentConfig, FeatureConfig
from directed_speech_detector.data.synthetic import generate_households
from directed_speech_detector.data.text import Vocabulary
from directed_speech_detector.features import LogMelExtractor, waveform_to_features
from directed_speech_detector.utils import get_logger

LOGGER = get_logger(__name__)
SPLIT_SEED_OFFSET = {"train": 0, "val": 1_000_003, "test": 2_000_003}
METADATA_KEYS = ("snr_db", "rt60", "level_db", "duration_s", "is_enrolled_speaker", "household")


@dataclass
class SplitArrays:
    """Ragged log-mel features for one split plus labels, transcripts and condition metadata."""

    feats: np.ndarray  # (sum_T, n_mels) float16
    offsets: np.ndarray  # (N + 1,) int64
    texts: list[str]
    labels: np.ndarray  # (N,) int64
    enroll_profile: np.ndarray  # (N, 2 * n_mels) float32
    meta: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, n: int) -> SplitArrays:
        end = int(self.offsets[n])
        return SplitArrays(
            self.feats[:end],
            self.offsets[: n + 1],
            self.texts[:n],
            self.labels[:n],
            self.enroll_profile[:n],
            {k: v[:n] for k, v in self.meta.items()},
        )


def _cache_key(data: DataConfig, feats: FeatureConfig, split: str, n: int) -> str:
    blob = json.dumps(
        {
            "data": {**data.__dict__, "num_workers": 0, "cache_dir": "", "manifest_dir": None},
            "feats": feats.__dict__,
            "split": split,
            "n": n,
            "v": 3,
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def build_split(data: DataConfig, feats_cfg: FeatureConfig, split: str, n: int) -> SplitArrays:
    """Generate ``n`` synthetic utterances for ``split`` and compute log-mel features."""
    extractor = LogMelExtractor(feats_cfg, data.sample_rate)
    households = generate_households(
        data.seed + SPLIT_SEED_OFFSET[split],
        n,
        data.utterances_per_household,
        data.max_seconds,
        data.num_workers,
    )
    chunks: list[np.ndarray] = []
    texts: list[str] = []
    labels: list[int] = []
    profiles: list[np.ndarray] = []
    meta: dict[str, list[float]] = {k: [] for k in METADATA_KEYS}
    for hh in households:
        _, enroll_profile = waveform_to_features(extractor, hh["enrollment"])
        for rec in hh["records"]:
            feats, _ = waveform_to_features(extractor, rec["wave"])
            chunks.append(feats.astype(np.float16))
            texts.append(rec["text"])
            labels.append(rec["label"])
            profiles.append(enroll_profile)
            meta["snr_db"].append(rec["snr_db"])
            meta["rt60"].append(rec["rt60"])
            meta["level_db"].append(rec["level_db"])
            meta["duration_s"].append(rec["wave"].size / data.sample_rate)
            meta["is_enrolled_speaker"].append(rec["is_enrolled_speaker"])
            meta["household"].append(hh["household"])
    lengths = np.array([c.shape[0] for c in chunks], dtype=np.int64)
    return SplitArrays(
        feats=np.concatenate(chunks, axis=0),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        texts=texts,
        labels=np.array(labels, dtype=np.int64),
        enroll_profile=np.stack(profiles).astype(np.float32),
        meta={k: np.array(v, dtype=np.float32) for k, v in meta.items()},
    )


def save_split(path: Path, split: SplitArrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {f"meta_{k}": v for k, v in split.meta.items()}
    arrays.update(
        feats=split.feats,
        offsets=split.offsets,
        texts=np.array(split.texts, dtype=str),
        labels=split.labels,
        enroll_profile=split.enroll_profile,
    )
    np.savez_compressed(path, **arrays)


def load_split(path: Path) -> SplitArrays:
    with np.load(path, allow_pickle=False) as z:
        return SplitArrays(
            feats=z["feats"],
            offsets=z["offsets"],
            texts=[str(t) for t in z["texts"]],
            labels=z["labels"],
            enroll_profile=z["enroll_profile"],
            meta={k[5:]: z[k] for k in z.files if k.startswith("meta_")},
        )


def get_split(cfg: ExperimentConfig, split: str) -> SplitArrays:
    """Load a split from the on-disk cache, generating it first if necessary."""
    if cfg.data.manifest_dir:
        from directed_speech_detector.data.manifest import split_from_manifest

        return split_from_manifest(
            Path(cfg.data.manifest_dir) / f"{split}.csv",
            cfg.features,
            cfg.data.max_seconds,
            cfg.data.sample_rate,
        )
    n = {"train": cfg.data.n_train, "val": cfg.data.n_val, "test": cfg.data.n_test}[split]
    path = Path(cfg.data.cache_dir) / f"{split}_{_cache_key(cfg.data, cfg.features, split, n)}.npz"
    if path.exists():
        return load_split(path)
    LOGGER.info("generating synthetic %s split (n=%d) -> %s", split, n, path)
    arrays = build_split(cfg.data, cfg.features, split, n)
    save_split(path, arrays)
    return arrays


class DirectedSpeechDataset(Dataset[dict[str, Any]]):
    """Map-style dataset yielding one utterance's features, tokens and label."""

    def __init__(
        self,
        split: SplitArrays,
        vocab: Vocabulary,
        max_text_len: int,
        word_dropout: float = 0.0,
        enrollment_dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.split, self.vocab, self.max_text_len = split, vocab, max_text_len
        self.word_dropout, self.enrollment_dropout = word_dropout, enrollment_dropout
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.split
        feats = s.feats[s.offsets[idx] : s.offsets[idx + 1]].astype(np.float32)
        ids = self.vocab.encode(s.texts[idx], self.max_text_len)
        if self.word_dropout > 0 and len(ids) > 1:
            kept = [i for i in ids if self._rng.random() >= self.word_dropout]
            ids = kept or ids[:1]
        has_enroll = 0.0 if self._rng.random() < self.enrollment_dropout else 1.0
        return {
            "feats": torch.from_numpy(feats),
            "tokens": torch.tensor(ids, dtype=torch.long),
            "enroll_profile": torch.from_numpy(s.enroll_profile[idx]),
            "has_enroll": has_enroll,
            "label": float(s.labels[idx]),
            "index": idx,
        }


def collate_batch(items: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad variable-length features/tokens and build masks (1 = valid)."""
    t_max = max(it["feats"].shape[0] for it in items)
    l_max = max(it["tokens"].shape[0] for it in items)
    n_mels = items[0]["feats"].shape[1]
    feats = torch.zeros(len(items), t_max, n_mels)
    feat_mask = torch.zeros(len(items), t_max)
    tokens = torch.zeros(len(items), l_max, dtype=torch.long)
    for i, it in enumerate(items):
        t, length = it["feats"].shape[0], it["tokens"].shape[0]
        feats[i, :t], feat_mask[i, :t] = it["feats"], 1.0
        tokens[i, :length] = it["tokens"]
    return {
        "feats": feats,
        "feat_mask": feat_mask,
        "tokens": tokens,
        "enroll_profile": torch.stack([it["enroll_profile"] for it in items]),
        "has_enroll": torch.tensor([it["has_enroll"] for it in items], dtype=torch.float32),
        "label": torch.tensor([it["label"] for it in items], dtype=torch.float32),
        "index": torch.tensor([it["index"] for it in items], dtype=torch.long),
    }


def make_loader(
    dataset: Dataset[dict[str, Any]], batch_size: int, shuffle: bool, seed: int = 0
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch, generator=generator
    )


def fit_feature_stats(split: SplitArrays) -> dict[str, torch.Tensor]:
    """Per-dimension mean/std of log-mel frames and of enrollment profiles (train split only)."""
    frames = split.feats.astype(np.float32)
    first = np.unique(split.meta["household"], return_index=True)[1]
    profiles = split.enroll_profile[first]
    return {
        "mel_mean": torch.from_numpy(frames.mean(0)),
        "mel_std": torch.from_numpy(frames.std(0) + 1e-5),
        "profile_mean": torch.from_numpy(profiles.mean(0)),
        "profile_std": torch.from_numpy(profiles.std(0) + 1e-5),
    }
