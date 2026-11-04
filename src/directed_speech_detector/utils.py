"""Shared utilities: seeding, devices, structured logging, timing, hardware info."""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from types import TracebackType
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record (structured logging)."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in self._RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, json_format: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            JsonFormatter() if json_format else logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class Timer:
    """Context manager measuring wall-clock milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def hardware_summary() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cores": __import__("os").cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
