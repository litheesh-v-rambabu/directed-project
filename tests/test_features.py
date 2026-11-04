import numpy as np
import torch

from directed_speech_detector.config import FeatureConfig
from directed_speech_detector.data.text import Vocabulary, normalize_text
from directed_speech_detector.features import (
    LogMelExtractor,
    mel_filterbank,
    pcm16_to_float,
    speaker_profile,
    waveform_to_features,
)


def test_logmel_shape_and_finite() -> None:
    cfg = FeatureConfig()
    extractor = LogMelExtractor(cfg)
    wave = torch.randn(2, 16_000)
    feats = extractor(wave)
    assert feats.shape == (2, extractor.num_frames(16_000), cfg.n_mels)
    assert torch.isfinite(feats).all()


def test_filterbank_is_nonnegative_and_covers_bands() -> None:
    fb = mel_filterbank(40, 400, 16_000, 20.0)
    assert fb.shape == (40, 201) and (fb >= 0).all() and (fb.sum(dim=1) > 0).all()


def test_tone_energy_lands_in_expected_band() -> None:
    extractor = LogMelExtractor(FeatureConfig())
    t = torch.arange(16_000) / 16_000
    low, high = (torch.sin(2 * torch.pi * f * t).unsqueeze(0) for f in (300.0, 3000.0))
    assert extractor(low)[0].mean(0).argmax() < extractor(high)[0].mean(0).argmax()


def test_speaker_profile_respects_mask() -> None:
    feats = torch.randn(1, 10, 4)
    mask = torch.tensor([[True] * 6 + [False] * 4])
    padded = feats.clone()
    padded[:, 6:] = 1e6
    assert torch.allclose(
        speaker_profile(feats[:, :6], torch.ones(1, 6, dtype=torch.bool)),
        speaker_profile(padded, mask),
        atol=1e-5,
    )


def test_waveform_to_features_returns_numpy() -> None:
    feats, profile = waveform_to_features(LogMelExtractor(FeatureConfig()), np.random.randn(8000))
    assert feats.dtype == np.float32 and profile.shape == (80,)


def test_pcm16_roundtrip() -> None:
    pcm = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    out = pcm16_to_float(pcm)
    assert np.allclose(out, [0, 0.5, -0.5, 32767 / 32768])


def test_vocabulary_encoding() -> None:
    vocab = Vocabulary.build(["Turn the lights on", "what's the weather"])
    assert vocab.encode("Turn ON the lights", 10) == [vocab.stoi[w] for w in ["turn", "on", "the", "lights"]]
    assert vocab.encode("zzz unseen", 10) == [1, 1]
    assert vocab.encode("   ", 10) == [1]
    assert len(vocab.encode("a b c d e f", 3)) == 3
    assert normalize_text("Hey, what's up?") == ["hey", "what's", "up"]
