"""Audio front-end: log-mel spectrograms and pooled speaker-profile statistics."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from directed_speech_detector.config import FeatureConfig


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int, f_min: float) -> torch.Tensor:
    """Triangular HTK mel filterbank of shape (n_mels, n_fft // 2 + 1)."""
    f_max = sample_rate / 2.0
    mel_points = np.linspace(_hz_to_mel(np.array(f_min)), _hz_to_mel(np.array(f_max)), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.linspace(0.0, f_max, n_fft // 2 + 1)
    fb = np.zeros((n_mels, bins.size), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (bins - lo) / max(mid - lo, 1e-8)
        down = (hi - bins) / max(hi - mid, 1e-8)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return torch.from_numpy(fb)


class LogMelExtractor(nn.Module):
    """Waveform (B, N) -> log-mel features (B, T, n_mels) with T = N // hop + 1."""

    def __init__(self, cfg: FeatureConfig, sample_rate: int = 16_000) -> None:
        super().__init__()
        self.cfg = cfg
        self.window: torch.Tensor
        self.filterbank: torch.Tensor
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)
        self.register_buffer(
            "filterbank",
            mel_filterbank(cfg.n_mels, cfg.n_fft, sample_rate, cfg.f_min),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            waveform,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            window=self.window,
            return_complex=True,
        )
        power = spec.abs().pow(2)  # (B, F, T)
        mel = torch.matmul(self.filterbank, power)  # (B, M, T)
        return torch.log(mel + 1e-8).transpose(1, 2).contiguous()

    def num_frames(self, num_samples: int) -> int:
        return num_samples // self.cfg.hop_length + 1


def speaker_profile(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked per-utterance mean and std of log-mel frames -> (B, 2 * n_mels).

    This lightweight statistics vector is the input to the speaker branch for both the
    enrollment clip and the evaluated utterance.
    """
    weights = mask.unsqueeze(-1).to(feats.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (feats * weights).sum(dim=1) / count
    var = (((feats - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / count
    return torch.cat([mean, torch.sqrt(var + 1e-6)], dim=-1)


def waveform_to_features(extractor: LogMelExtractor, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single waveform -> (log-mel (T, M), speaker profile (2M,)) as float32 arrays."""
    wave = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)
    feats = extractor(wave)
    mask = torch.ones(feats.shape[:2], dtype=torch.bool)
    return feats[0].numpy(), speaker_profile(feats, mask)[0].numpy()


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Decode little-endian 16-bit mono PCM bytes to float32 in [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def frames_for_seconds(seconds: float, hop_length: int, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate / hop_length)) + 1
