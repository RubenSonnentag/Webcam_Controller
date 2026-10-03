from __future__ import annotations

import time

from .gesture import GestureEngine
from .metrics import RollingMetrics
from .types import FaceSignals


def run_synthetic(seconds: float = 5.0, rate: float = 60.0) -> dict[str, object]:
    engine = GestureEngine()
    metrics = RollingMetrics()
    start_ns = time.perf_counter_ns()
    period_s = 1.0 / rate
    count = 0
    events = 0
    while (time.perf_counter_ns() - start_ns) / 1e9 < seconds:
        cycle_ms = ((time.perf_counter_ns() - start_ns) / 1e6) % 1000.0
        left = 0.9 if 300.0 <= cycle_ms <= 500.0 else 0.05
        timestamp_ns = time.perf_counter_ns()
        before_ns = time.perf_counter_ns()
        emitted = engine.update(
            FaceSignals(
                timestamp_ns=timestamp_ns,
                face_present=True,
                confidence=1.0,
                person_left_eye_closed=left,
                person_right_eye_closed=0.05,
            )
        )
        after_ns = time.perf_counter_ns()
        metrics.add("gesture_ms", (after_ns - before_ns) / 1e6)
        events += len(emitted)
        count += 1
        target = start_ns + int(count * period_s * 1e9)
        remaining = (target - time.perf_counter_ns()) / 1e9
        if remaining > 0:
            time.sleep(remaining)
    return {
        "output": "Disabled",
        "samples": count,
        "events": events,
        "effective_rate": count / max(seconds, 1e-9),
        "metrics_ms": metrics.snapshot(),
    }
