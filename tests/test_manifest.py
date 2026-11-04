import wave

import numpy as np
import pytest

from directed_speech_detector.config import FeatureConfig, load_config
from directed_speech_detector.data.dataset import get_split
from directed_speech_detector.data.manifest import read_manifest, read_wav, split_from_manifest


def _write_wav(path, seconds=1.0, rate=16_000, channels=1):
    samples = (np.random.default_rng(0).normal(0, 0.1, int(seconds * rate) * channels) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


@pytest.fixture()
def manifest(tmp_path):
    for i in range(3):
        _write_wav(tmp_path / f"{i}.wav", seconds=0.5 + i * 0.25)
    (tmp_path / "test.csv").write_text(
        "audio_path,transcript,label\n0.wav,turn on the lights,1\n1.wav,did you see the game,0\n2.wav,play jazz,1\n"
    )
    return tmp_path / "test.csv"


def test_read_manifest_and_wav(manifest) -> None:
    rows = read_manifest(manifest)
    assert [r.label for r in rows] == [1, 0, 1]
    assert read_wav(rows[0].audio_path).size == 8000


def test_split_from_manifest(manifest) -> None:
    split = split_from_manifest(manifest, FeatureConfig())
    assert len(split) == 3 and split.feats.shape[1] == 40
    assert np.allclose(split.meta["duration_s"], [0.5, 0.75, 1.0])
    assert not split.enroll_profile.any()


def test_rejects_wrong_audio_format(tmp_path) -> None:
    _write_wav(tmp_path / "stereo.wav", channels=2)
    with pytest.raises(ValueError, match="mono 16-bit"):
        read_wav(tmp_path / "stereo.wav")


def test_config_routes_to_manifest_and_rejects_speaker(manifest) -> None:
    cfg = load_config("configs/real_audio_template.yaml", [f"data.manifest_dir={manifest.parent}"])
    assert len(get_split(cfg, "test")) == 3
    with pytest.raises(ValueError, match="enrollment"):
        load_config("configs/real_audio_template.yaml", ["model.modalities=[audio, speaker]"])
