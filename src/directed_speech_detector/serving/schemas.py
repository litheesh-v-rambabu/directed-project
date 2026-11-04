"""Request/response models and audio validation for the serving API."""

from __future__ import annotations

import base64
import binascii

import numpy as np
from pydantic import BaseModel, Field

from directed_speech_detector.features import pcm16_to_float

SAMPLE_RATE = 16_000
MIN_SECONDS, MAX_SECONDS = 0.2, 10.0


class AudioError(ValueError):
    """Raised when a request's audio payload is malformed."""


class PredictRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 of little-endian 16-bit mono PCM audio")
    sample_rate: int = Field(SAMPLE_RATE, description="Must be 16000")
    transcript: str = Field(..., min_length=1, max_length=500, description="ASR transcript of the audio")
    enrollment_audio_base64: str | None = Field(
        None, description="Optional enrolled-user voice clip, same format"
    )


class Latency(BaseModel):
    feature_ms: float
    model_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    probability_directed: float = Field(..., ge=0.0, le=1.0)
    is_directed: bool
    threshold: float
    backend: str
    enrollment_used: bool
    latency: Latency


def decode_audio(payload: str, field: str) -> np.ndarray:
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AudioError(f"{field} is not valid base64") from exc
    if len(raw) % 2:
        raise AudioError(f"{field} must contain whole 16-bit samples")
    audio = pcm16_to_float(raw)
    seconds = audio.size / SAMPLE_RATE
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise AudioError(f"{field} must be {MIN_SECONDS}-{MAX_SECONDS} s long, got {seconds:.2f} s")
    return audio


def encode_audio(audio: np.ndarray) -> str:
    """Float waveform in [-1, 1] -> base64 PCM16 (client helper used by tests and docs)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")
