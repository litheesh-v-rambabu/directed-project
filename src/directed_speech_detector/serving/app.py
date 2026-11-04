"""FastAPI service: /predict, /health, /metrics with validation, structured logs and latency tracking."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from directed_speech_detector.serving.inference import Detector
from directed_speech_detector.serving.metrics import MetricsRegistry
from directed_speech_detector.serving.schemas import (
    SAMPLE_RATE,
    AudioError,
    Latency,
    PredictRequest,
    PredictResponse,
    decode_audio,
)
from directed_speech_detector.utils import get_logger

LOGGER = get_logger("dsd.api", json_format=True)


def create_app(detector: Detector) -> FastAPI:
    app = FastAPI(title="Directed Speech Detector", version="0.1.0")
    registry = MetricsRegistry()

    @app.middleware("http")
    async def track(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            registry.observe_request(request.url.path, status, elapsed)
            LOGGER.info(
                "request",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": round(elapsed, 2),
                },
            )

    @app.exception_handler(AudioError)
    async def audio_error_handler(_: Request, exc: AudioError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": detector.info()}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return registry.render()

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if req.sample_rate != SAMPLE_RATE:
            raise HTTPException(status_code=422, detail=f"sample_rate must be {SAMPLE_RATE}")
        start = time.perf_counter()
        audio = decode_audio(req.audio_base64, "audio_base64")
        enrollment = (
            decode_audio(req.enrollment_audio_base64, "enrollment_audio_base64")
            if req.enrollment_audio_base64
            else None
        )
        pred = detector.predict(audio, req.transcript, enrollment)
        registry.observe_decision(pred.is_directed)
        return PredictResponse(
            probability_directed=pred.probability,
            is_directed=pred.is_directed,
            threshold=pred.threshold,
            backend=detector.backend,
            enrollment_used=pred.enrollment_used,
            latency=Latency(
                feature_ms=pred.feature_ms,
                model_ms=pred.model_ms,
                total_ms=(time.perf_counter() - start) * 1000.0,
            ),
        )

    return app


def create_app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory``: reads DSD_RUN_DIR and DSD_BACKEND."""
    run_dir = os.environ.get("DSD_RUN_DIR", "artifacts/model")
    return create_app(Detector.from_run_dir(run_dir, os.environ.get("DSD_BACKEND", "fp32")))
