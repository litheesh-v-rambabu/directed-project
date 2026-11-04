FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image small; the serving extras add FastAPI + ONNX Runtime.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,export]"

# A trained run directory (model.pt, calibration.json, export/model.onnx) is mounted here.
ENV DSD_RUN_DIR=/models/run DSD_BACKEND=fp32
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "directed_speech_detector.serving.app:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
