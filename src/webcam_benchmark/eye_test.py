from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from .backends.base import FaceBackend
from .capture import CameraCapture
from .config import CameraConfig
from .latest_frame import LatestFrameBuffer
from .types import FaceSignals


WINDOW_TITLE = "Phase-2-Augentest (keine Ausgabe)"


@dataclass(frozen=True, slots=True)
class EyeStep:
    name: str
    prompt: str
    duration_s: float


STEPS = (
    EyeStep("neutral", "Entspannt schauen; spontane Blinzler sind erlaubt", 10.0),
    EyeStep("normal_blinks", "8-10 Mal normal blinzeln, mit kurzen Pausen", 15.0),
    EyeStep("left_repeats", "LINKES Auge 5 Mal je ca. 0,5 s schliessen", 12.0),
    EyeStep("right_repeats", "RECHTES Auge 5 Mal je ca. 0,5 s schliessen", 12.0),
)


def run_eye_test(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    export_path: Path | None,
    prepare_s: float = 3.0,
) -> dict[str, object]:
    import cv2

    samples: dict[str, list[FaceSignals]] = {step.name: [] for step in STEPS}
    buffer = LatestFrameBuffer()
    capture = CameraCapture(camera_config, buffer)
    aborted = False
    capture.start()
    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1280, 720)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
        for index, step in enumerate(STEPS):
            if not _await_space(buffer, capture, step, index):
                aborted = True
                break
            if not _countdown(buffer, capture, step, index, prepare_s):
                aborted = True
                break
            started = time.perf_counter()
            while time.perf_counter() - started < step.duration_s:
                frame = buffer.take(0.25)
                if capture.error:
                    raise capture.error
                if frame is None:
                    continue
                signal = backend.analyze(frame.image_bgr, frame.arrived_ns)
                samples[step.name].append(signal)
                remaining = step.duration_s - (time.perf_counter() - started)
                if _show(frame.image_bgr, "MESSUNG", step, index, remaining, signal) == 27:
                    aborted = True
                    break
            if aborted:
                break
    finally:
        capture.stop()
        backend.close()
        cv2.destroyAllWindows()

    report: dict[str, object] = {
        "privacy": "Aggregate timing and signal statistics only; no images/video stored",
        "output": "Disabled",
        "aborted": aborted,
        "camera": {
            "requested": [camera_config.width, camera_config.height, camera_config.fps],
            "actual": [capture.actual_width, capture.actual_height, capture.actual_fps],
            "requested_fourcc": camera_config.fourcc.upper(),
            "actual_fourcc": capture.actual_fourcc,
            "requested_exposure": camera_config.exposure,
            "actual_exposure": capture.actual_exposure,
        },
        "thresholds": {
            "eye_close": 0.35,
            "left_asymmetry": 0.13,
            "right_asymmetry": 0.09,
        },
        "steps": {name: _eye_summary(values) for name, values in samples.items()},
    }
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _eye_summary(samples: list[FaceSignals]) -> dict[str, object]:
    present = [sample for sample in samples if sample.face_present]
    left_closures = _intervals(present, lambda s: s.person_left_eye_closed >= 0.35)
    right_closures = _intervals(present, lambda s: s.person_right_eye_closed >= 0.35)
    left_unilateral = _intervals(
        present,
        lambda s: s.person_left_eye_closed >= 0.35
        and s.person_left_eye_closed - s.person_right_eye_closed >= 0.13,
    )
    right_unilateral = _intervals(
        present,
        lambda s: s.person_right_eye_closed >= 0.35
        and s.person_right_eye_closed - s.person_left_eye_closed >= 0.09,
    )
    onset_deltas = _paired_onset_deltas(left_closures, right_closures)
    span_s = (
        (present[-1].timestamp_ns - present[0].timestamp_ns) / 1e9
        if len(present) > 1
        else 0.0
    )
    return {
        "frames": len(samples),
        "face_present_rate": len(present) / max(len(samples), 1),
        "left_closures": _duration_stats(left_closures),
        "right_closures": _duration_stats(right_closures),
        "paired_blinks": {
            "count": len(onset_deltas),
            "onset_delta_ms": _stats(onset_deltas),
        },
        "left_unilateral": _duration_stats(left_unilateral, span_s),
        "right_unilateral": _duration_stats(right_unilateral, span_s),
    }


def _intervals(samples: list[FaceSignals], predicate: object) -> list[tuple[int, int]]:
    if not samples:
        return []
    intervals: list[tuple[int, int]] = []
    started: int | None = None
    periods = [b.timestamp_ns - a.timestamp_ns for a, b in zip(samples, samples[1:])]
    tail = int(median(periods)) if periods else 0
    for sample in samples:
        active = bool(predicate(sample))  # type: ignore[operator]
        if active and started is None:
            started = sample.timestamp_ns
        elif not active and started is not None:
            intervals.append((started, sample.timestamp_ns))
            started = None
    if started is not None:
        intervals.append((started, samples[-1].timestamp_ns + tail))
    return intervals


def _paired_onset_deltas(
    left: list[tuple[int, int]], right: list[tuple[int, int]], max_delta_ms: float = 150.0
) -> list[float]:
    unused = set(range(len(right)))
    deltas: list[float] = []
    for left_start, _ in left:
        candidates = [(abs(left_start - right[i][0]), i) for i in unused]
        if not candidates:
            continue
        delta_ns, index = min(candidates)
        if delta_ns / 1e6 <= max_delta_ms:
            deltas.append(delta_ns / 1e6)
            unused.remove(index)
    return deltas


def _duration_stats(intervals: list[tuple[int, int]], span_s: float = 0.0) -> dict[str, object]:
    durations = [(end - start) / 1e6 for start, end in intervals]
    result: dict[str, object] = {"count": len(intervals), "duration_ms": _stats(durations)}
    if span_s > 0:
        result["events_per_minute"] = len(intervals) * 60.0 / span_s
    return result


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _await_space(buffer: LatestFrameBuffer, capture: CameraCapture, step: EyeStep, index: int) -> bool:
    while True:
        frame = buffer.take(0.25)
        if capture.error:
            raise capture.error
        if frame is None:
            continue
        key = _show(frame.image_bgr, "BEREIT? LEERTASTE DRUECKEN", step, index, 0.0)
        if key == 27:
            return False
        if key == 32:
            return True


def _countdown(
    buffer: LatestFrameBuffer, capture: CameraCapture, step: EyeStep, index: int, seconds: float
) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        frame = buffer.take(0.25)
        if capture.error:
            raise capture.error
        if frame is None:
            continue
        remaining = seconds - (time.perf_counter() - started)
        if _show(frame.image_bgr, "START IN", step, index, remaining) == 27:
            return False
    return True


def _show(
    image: object,
    phase: str,
    step: EyeStep,
    index: int,
    remaining: float,
    signal: FaceSignals | None = None,
) -> int:
    import cv2

    display = image.copy()
    cv2.rectangle(display, (0, 0), (display.shape[1], 150), (0, 0, 0), -1)
    lines = (
        f"{phase}  {index + 1}/{len(STEPS)}  {max(remaining, 0):.1f}s",
        step.prompt,
        "Leertaste = Start | Esc = Abbruch | KEINE AUSGABE",
    )
    for row, line in enumerate(lines):
        cv2.putText(display, line, (24, 38 + row * 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    if signal is not None:
        text = f"Linkes/rechtes Auge: {signal.person_left_eye_closed:.2f} / {signal.person_right_eye_closed:.2f}"
        cv2.putText(display, text, (24, display.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow(WINDOW_TITLE, display)
    return cv2.waitKey(1) & 0xFF
