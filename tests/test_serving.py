import numpy as np
import pytest
from fastapi.testclient import TestClient

from directed_speech_detector.serving.app import create_app
from directed_speech_detector.serving.inference import Detector
from directed_speech_detector.serving.schemas import encode_audio


@pytest.fixture()
def client(random_model, tiny_cfg, vocab) -> TestClient:
    return TestClient(create_app(Detector(random_model, tiny_cfg, vocab, temperature=1.2, threshold=0.4)))


def _audio(seconds: float = 1.0) -> str:
    return encode_audio(np.random.default_rng(0).normal(0, 0.05, int(16_000 * seconds)))


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"]["modalities"] == ["audio", "text", "speaker"]


def test_predict_success_and_enrollment(client) -> None:
    payload = {"audio_base64": _audio(), "transcript": "turn off the kitchen lights"}
    base = client.post("/predict", json=payload)
    assert base.status_code == 200
    body = base.json()
    assert 0.0 <= body["probability_directed"] <= 1.0 and body["threshold"] == 0.4
    assert body["is_directed"] == (body["probability_directed"] > 0.4)
    assert body["enrollment_used"] is False and body["latency"]["total_ms"] >= body["latency"]["model_ms"]
    enrolled = client.post("/predict", json={**payload, "enrollment_audio_base64": _audio(2.0)}).json()
    assert enrolled["enrollment_used"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"audio_base64": "!!notbase64!!", "transcript": "hi"},
        {"audio_base64": "AAAA", "transcript": "hi"},  # 3 bytes -> odd sample count
        {"audio_base64": "A" * 8, "transcript": "hi"},  # too short
        {"audio_base64": "", "transcript": ""},
        {"transcript": "missing audio"},
        {"audio_base64": "AAAA", "transcript": "hi", "sample_rate": 8000},
    ],
)
def test_predict_rejects_invalid_requests(client, payload) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_rejects_overlong_audio_and_bad_enrollment(client) -> None:
    assert client.post("/predict", json={"audio_base64": _audio(11.0), "transcript": "hi"}).status_code == 422
    bad = {"audio_base64": _audio(), "transcript": "hi", "enrollment_audio_base64": "###"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_endpoint_counts_requests(client) -> None:
    client.post("/predict", json={"audio_base64": _audio(), "transcript": "play jazz"})
    client.post("/predict", json={"transcript": "bad"})
    text = client.get("/metrics").text
    assert 'dsd_requests_total{endpoint="/predict",status="200"} 1' in text
    assert 'dsd_requests_total{endpoint="/predict",status="422"} 1' in text
    assert "dsd_predict_latency_ms_count 1" in text and 'quantile="0.95"' in text


def test_request_id_header_roundtrip(client) -> None:
    assert client.get("/health", headers={"x-request-id": "abc123"}).headers["x-request-id"] == "abc123"
