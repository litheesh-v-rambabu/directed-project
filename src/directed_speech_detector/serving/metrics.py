"""Minimal thread-safe Prometheus-format metrics (no external dependency)."""

from __future__ import annotations

import threading
from collections import defaultdict, deque

import numpy as np

LATENCY_BUCKETS_MS = (1, 2, 5, 10, 20, 50, 100, 250, 500, 1000)


class MetricsRegistry:
    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, int], int] = defaultdict(int)
        self._decisions: dict[str, int] = defaultdict(int)
        self._bucket_counts = [0] * (len(LATENCY_BUCKETS_MS) + 1)
        self._latency_sum = 0.0
        self._latency_count = 0
        self._recent: deque[float] = deque(maxlen=window)

    def observe_request(self, endpoint: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self._requests[(endpoint, status)] += 1
            if endpoint == "/predict" and status == 200:
                self._latency_sum += latency_ms
                self._latency_count += 1
                self._recent.append(latency_ms)
                idx = next(
                    (i for i, b in enumerate(LATENCY_BUCKETS_MS) if latency_ms <= b), len(LATENCY_BUCKETS_MS)
                )
                self._bucket_counts[idx] += 1

    def observe_decision(self, directed: bool) -> None:
        with self._lock:
            self._decisions["directed" if directed else "background"] += 1

    def quantile(self, q: float) -> float:
        with self._lock:
            return float(np.percentile(list(self._recent), q)) if self._recent else 0.0

    def render(self) -> str:
        with self._lock:
            lines = ["# TYPE dsd_requests_total counter"]
            lines += [
                f'dsd_requests_total{{endpoint="{e}",status="{s}"}} {n}'
                for (e, s), n in sorted(self._requests.items())
            ]
            lines.append("# TYPE dsd_decisions_total counter")
            lines += [
                f'dsd_decisions_total{{decision="{d}"}} {n}' for d, n in sorted(self._decisions.items())
            ]
            lines.append("# TYPE dsd_predict_latency_ms histogram")
            cumulative = 0
            for bound, count in zip(LATENCY_BUCKETS_MS, self._bucket_counts, strict=False):
                cumulative += count
                lines.append(f'dsd_predict_latency_ms_bucket{{le="{bound}"}} {cumulative}')
            lines.append(f'dsd_predict_latency_ms_bucket{{le="+Inf"}} {self._latency_count}')
            lines.append(f"dsd_predict_latency_ms_sum {self._latency_sum:.3f}")
            lines.append(f"dsd_predict_latency_ms_count {self._latency_count}")
            recent = list(self._recent)
        for q in (50, 95):
            value = float(np.percentile(recent, q)) if recent else 0.0
            lines.append(f'dsd_predict_latency_ms_window{{quantile="0.{q}"}} {value:.3f}')
        return "\n".join(lines) + "\n"
