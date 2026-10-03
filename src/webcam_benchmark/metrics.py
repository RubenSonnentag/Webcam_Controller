from __future__ import annotations

import math
import statistics
import threading
from collections import defaultdict, deque


class RollingMetrics:
    def __init__(self, capacity: int = 3600) -> None:
        self._values: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=capacity)
        )
        self._lock = threading.Lock()

    def add(self, name: str, value: float) -> None:
        if math.isfinite(value):
            with self._lock:
                self._values[name].append(float(value))

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            copied = {key: list(values) for key, values in self._values.items()}
        return {key: summarize(values) for key, values in copied.items()}


def percentile(sorted_values: list[float], percent: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)
    return {
        "count": float(len(ordered)),
        "median": statistics.median(ordered),
        "p95": percentile(ordered, 95.0),
        "p99": percentile(ordered, 99.0),
        "min": ordered[0],
        "max": ordered[-1],
    }
