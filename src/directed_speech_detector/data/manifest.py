"""Real-audio data path: build a split from a CSV manifest of 16 kHz mono 16-bit WAV files.

Manifest columns: ``audio_path,transcript,label`` (label 1 = directed, 0 = background). Relative
paths resolve against the manifest's directory. Real corpora have no enrolled-speaker profile, so
the speaker modality is unavailable on this path (enrollment profile is zero and ``has_enroll`` 0).
"""

from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from directed_speech_detector.config import FeatureConfig
from directed_speech_detector.data.dataset import METADATA_KEYS, SplitArrays
from directed_speech_detector.features import LogMelExtractor, waveform_to_features


@dataclass(frozen=True)
class ManifestRow:
    audio_path: Path
    transcript: str
    label: int


def read_wav(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    """Read a mono 16-bit PCM WAV with the standard library; reject anything else explicitly."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != sample_rate:
            raise ValueError(f"{path}: expected mono 16-bit PCM at {sample_rate} Hz")
        pcm = handle.readframes(handle.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def read_manifest(path: Path) -> list[ManifestRow]:
    rows = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            label = int(record["label"])
            if label not in (0, 1):
                raise ValueError(f"label must be 0 or 1, got {label}")
            rows.append(ManifestRow(path.parent / record["audio_path"], record["transcript"], label))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def split_from_manifest(
    path: Path, features: FeatureConfig, max_seconds: float = 4.0, sample_rate: int = 16_000
) -> SplitArrays:
    extractor = LogMelExtractor(features, sample_rate)
    chunks, texts, labels, durations = [], [], [], []
    for row in read_manifest(path):
        wave_data = read_wav(row.audio_path, sample_rate)[: int(max_seconds * sample_rate)]
        feats, _ = waveform_to_features(extractor, wave_data)
        chunks.append(feats.astype(np.float16))
        texts.append(row.transcript)
        labels.append(row.label)
        durations.append(wave_data.size / sample_rate)
    n = len(chunks)
    lengths = np.array([c.shape[0] for c in chunks], dtype=np.int64)
    meta = {k: np.full(n, np.nan, dtype=np.float32) for k in METADATA_KEYS}
    meta["duration_s"] = np.array(durations, dtype=np.float32)
    meta["is_enrolled_speaker"] = np.zeros(n, dtype=np.float32)
    meta["household"] = np.zeros(n, dtype=np.float32)
    return SplitArrays(
        feats=np.concatenate(chunks),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        texts=texts,
        labels=np.array(labels, dtype=np.int64),
        enroll_profile=np.zeros((n, 2 * features.n_mels), dtype=np.float32),
        meta=meta,
    )
