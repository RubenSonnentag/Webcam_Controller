from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends.base import FaceBackend
from .capture import CameraCapture
from .config import CameraConfig
from .eye_test import _intervals, _paired_onset_deltas
from .latest_frame import LatestFrameBuffer
from .types import FaceSignals


WINDOW_TITLE = "Kontrollierter Augen-Einzeltest (keine Ausgabe)"
DURATIONS_MS = (90.0, 100.0, 120.0, 150.0)
EAR_CLOSE_THRESHOLDS = (0.40, 0.425, 0.45)


@dataclass(frozen=True, slots=True)
class Trial:
    expected: str
    prompt: str
    number: int
    total: int = 10


def _trials(count: int = 10, selected_groups: tuple[str, ...] | None = None) -> list[Trial]:
    groups = (
        ("normal_blink", "Genau EINMAL normal blinzeln"),
        ("left_eye", "Genau EINMAL das LINKE Auge schliessen"),
        ("right_eye", "Genau EINMAL das RECHTE Auge schliessen"),
    )
    if selected_groups is not None:
        groups = tuple(group for group in groups if group[0] in selected_groups)
    return [
        Trial(expected, prompt, number, count)
        for expected, prompt in groups
        for number in range(1, count + 1)
    ]


def run_trial_test(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    export_path: Path | None,
    countdown_s: float = 1.5,
    measurement_s: float = 2.5,
    trials_per_gesture: int = 10,
    trial_groups: tuple[str, ...] | None = None,
) -> dict[str, object]:
    import cv2

    buffer = LatestFrameBuffer()
    capture = CameraCapture(camera_config, buffer)
    results: list[dict[str, object]] = []
    aborted = False
    if trials_per_gesture < 1:
        raise ValueError("trials_per_gesture must be positive")
    trials = _trials(trials_per_gesture, trial_groups)
    if not trials:
        raise ValueError("At least one valid trial group is required")
    capture.start()
    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1280, 720)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
        for index, trial in enumerate(trials):
            if not _await_space(buffer, capture, trial, index, len(trials)):
                aborted = True
                break
            if not _countdown(buffer, capture, trial, index, len(trials), countdown_s):
                aborted = True
                break
            samples: list[FaceSignals] = []
            started = time.perf_counter()
            while time.perf_counter() - started < measurement_s:
                frame = buffer.take(0.25)
                if capture.error:
                    raise capture.error
                if frame is None:
                    continue
                signal = backend.analyze(frame.image_bgr, frame.arrived_ns)
                samples.append(signal)
                remaining = measurement_s - (time.perf_counter() - started)
                if _show(frame.image_bgr, "JETZT", trial, index, len(trials), remaining, signal) == 27:
                    aborted = True
                    break
            results.append(_classify_trial(trial, samples))
            if aborted:
                break
    finally:
        capture.stop()
        backend.close()
        cv2.destroyAllWindows()

    report: dict[str, object] = {
        "privacy": "Per-trial classifications and aggregate timing only; no images/video stored",
        "output": "Disabled",
        "aborted": aborted,
        "camera": {
            "requested": [camera_config.width, camera_config.height, camera_config.fps],
            "actual": [capture.actual_width, capture.actual_height, capture.actual_fps],
        },
        "thresholds": {
            "eye_close": 0.35,
            "left_asymmetry": 0.13,
            "right_asymmetry": 0.09,
            "bilateral_onset_delta_ms": 50.0,
            "minimum_durations_ms": list(DURATIONS_MS),
        },
        "summary": _summarize_trials(results),
        "ear_summary": _summarize_trials(results, decision_field="ear_decisions"),
        "ear_threshold_summaries": {
            _threshold_key(threshold): _summarize_trials(
                results,
                decision_field=f"ear_decisions_{_threshold_key(threshold)}",
            )
            for threshold in EAR_CLOSE_THRESHOLDS
        },
        "trials": results,
    }
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _classify_trial(trial: Trial, samples: list[FaceSignals]) -> dict[str, object]:
    present = [sample for sample in samples if sample.face_present]
    left = _intervals(
        present,
        lambda s: s.person_left_eye_closed >= 0.35
        and s.person_left_eye_closed - s.person_right_eye_closed >= 0.13,
    )
    right = _intervals(
        present,
        lambda s: s.person_right_eye_closed >= 0.35
        and s.person_right_eye_closed - s.person_left_eye_closed >= 0.09,
    )
    left_all = _intervals(present, lambda s: s.person_left_eye_closed >= 0.35)
    right_all = _intervals(present, lambda s: s.person_right_eye_closed >= 0.35)
    bilateral = _paired_onset_deltas(left_all, right_all, 50.0)
    left = _merge_short_gaps(left, 70.0)
    right = _merge_short_gaps(right, 70.0)
    left_max = max(((end - start) / 1e6 for start, end in left), default=0.0)
    right_max = max(((end - start) / 1e6 for start, end in right), default=0.0)
    ear = _ear_trial_summary(present)
    ear_candidates = {
        threshold: _ear_candidate_durations(present, threshold)
        for threshold in EAR_CLOSE_THRESHOLDS
    }
    ear_left_max, ear_right_max = ear_candidates[0.45]
    decisions = {}
    ear_decisions = {}
    ear_decisions_by_threshold: dict[float, dict[str, str]] = {
        threshold: {} for threshold in EAR_CLOSE_THRESHOLDS
    }
    for duration in DURATIONS_MS:
        detected_left = left_max >= duration
        detected_right = right_max >= duration
        if detected_left and detected_right:
            detected = "ambiguous"
        elif detected_left:
            detected = "left_eye"
        elif detected_right:
            detected = "right_eye"
        elif bilateral:
            detected = "normal_blink"
        else:
            detected = "none"
        decisions[str(int(duration))] = detected
        ear_left = ear_left_max >= duration
        ear_right = ear_right_max >= duration
        if ear_left and ear_right:
            ear_detected = "ambiguous"
        elif ear_left:
            ear_detected = "left_eye"
        elif ear_right:
            ear_detected = "right_eye"
        elif bilateral:
            ear_detected = "normal_blink"
        else:
            ear_detected = "none"
        ear_decisions[str(int(duration))] = ear_detected
        for close_threshold, (threshold_left_max, threshold_right_max) in ear_candidates.items():
            threshold_left = threshold_left_max >= duration
            threshold_right = threshold_right_max >= duration
            if threshold_left and threshold_right:
                threshold_detected = "ambiguous"
            elif threshold_left:
                threshold_detected = "left_eye"
            elif threshold_right:
                threshold_detected = "right_eye"
            elif bilateral:
                threshold_detected = "normal_blink"
            else:
                threshold_detected = "none"
            ear_decisions_by_threshold[close_threshold][str(int(duration))] = threshold_detected
    return {
        "expected": trial.expected,
        "number": trial.number,
        "face_present_rate": len(present) / max(len(samples), 1),
        "left_candidate_max_ms": left_max,
        "right_candidate_max_ms": right_max,
        "paired_blink_count": len(bilateral),
        "eye_aspect_ratio": ear,
        "ear_left_candidate_max_ms": ear_left_max,
        "ear_right_candidate_max_ms": ear_right_max,
        "ear_decisions": ear_decisions,
        **{
            f"ear_decisions_{_threshold_key(threshold)}": decisions
            for threshold, decisions in ear_decisions_by_threshold.items()
        },
        "decisions": decisions,
    }


def _ear_trial_summary(samples: list[FaceSignals]) -> dict[str, object] | None:
    if not samples:
        return None
    left = np.asarray([sample.person_left_eye_aspect_ratio for sample in samples], dtype=float)
    right = np.asarray([sample.person_right_eye_aspect_ratio for sample in samples], dtype=float)
    if not np.any(left > 0) or not np.any(right > 0):
        return None
    left_open = float(np.percentile(left, 90))
    right_open = float(np.percentile(right, 90))
    left_closedness = 1.0 - left / max(left_open, 1e-9)
    right_closedness = 1.0 - right / max(right_open, 1e-9)
    asymmetry = left_closedness - right_closedness
    return {
        "left_open_p90": left_open,
        "right_open_p90": right_open,
        "left_min": float(np.min(left)),
        "right_min": float(np.min(right)),
        "left_closedness_max": float(np.max(left_closedness)),
        "right_closedness_max": float(np.max(right_closedness)),
        "closedness_asymmetry_min": float(np.min(asymmetry)),
        "closedness_asymmetry_max": float(np.max(asymmetry)),
    }


def _ear_candidate_durations(
    samples: list[FaceSignals], close_threshold: float = 0.45
) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    left_values = np.asarray([sample.person_left_eye_aspect_ratio for sample in samples], dtype=float)
    right_values = np.asarray([sample.person_right_eye_aspect_ratio for sample in samples], dtype=float)
    if not np.any(left_values > 0) or not np.any(right_values > 0):
        return 0.0, 0.0
    left_open = float(np.percentile(left_values, 90))
    right_open = float(np.percentile(right_values, 90))
    left_closed = 1.0 - left_values / max(left_open, 1e-9)
    right_closed = 1.0 - right_values / max(right_open, 1e-9)
    indexed = list(zip(samples, left_closed, right_closed))
    left_intervals = _numeric_intervals(
        indexed, lambda left, right: left >= close_threshold and left - right >= 0.23
    )
    right_intervals = _numeric_intervals(
        indexed, lambda left, right: right >= close_threshold and right - left >= 0.22
    )
    left_intervals = _merge_short_gaps(left_intervals, 70.0)
    right_intervals = _merge_short_gaps(right_intervals, 70.0)
    left_max = max(((end - start) / 1e6 for start, end in left_intervals), default=0.0)
    right_max = max(((end - start) / 1e6 for start, end in right_intervals), default=0.0)
    return left_max, right_max


def _threshold_key(threshold: float) -> str:
    return f"close_{threshold:.3f}"


def _numeric_intervals(
    samples: list[tuple[FaceSignals, float, float]], predicate: object
) -> list[tuple[int, int]]:
    if not samples:
        return []
    intervals: list[tuple[int, int]] = []
    started: int | None = None
    timestamps = [sample.timestamp_ns for sample, _, _ in samples]
    periods = [b - a for a, b in zip(timestamps, timestamps[1:])]
    tail = int(np.median(periods)) if periods else 0
    for sample, left, right in samples:
        active = bool(predicate(left, right))  # type: ignore[operator]
        if active and started is None:
            started = sample.timestamp_ns
        elif not active and started is not None:
            intervals.append((started, sample.timestamp_ns))
            started = None
    if started is not None:
        intervals.append((started, samples[-1][0].timestamp_ns + tail))
    return intervals


def _merge_short_gaps(intervals: list[tuple[int, int]], max_gap_ms: float) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    max_gap_ns = int(max_gap_ms * 1e6)
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap_ns:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _summarize_trials(
    results: list[dict[str, object]], *, decision_field: str = "decisions"
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for duration in DURATIONS_MS:
        key = str(int(duration))
        matrix: dict[str, dict[str, int]] = {}
        for result in results:
            expected = str(result["expected"])
            detected = str(result[decision_field][key])  # type: ignore[index]
            matrix.setdefault(expected, {})[detected] = matrix.setdefault(expected, {}).get(detected, 0) + 1
        summary[key] = matrix
    return summary


def _await_space(
    buffer: LatestFrameBuffer, capture: CameraCapture, trial: Trial, index: int, total: int
) -> bool:
    while True:
        frame = buffer.take(0.25)
        if capture.error:
            raise capture.error
        if frame is None:
            continue
        key = _show(frame.image_bgr, "BEREIT? LEERTASTE", trial, index, total, 0.0)
        if key == 27:
            return False
        if key == 32:
            return True


def _countdown(
    buffer: LatestFrameBuffer,
    capture: CameraCapture,
    trial: Trial,
    index: int,
    total: int,
    seconds: float,
) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        frame = buffer.take(0.25)
        if capture.error:
            raise capture.error
        if frame is None:
            continue
        if _show(
            frame.image_bgr,
            "START IN",
            trial,
            index,
            total,
            seconds - (time.perf_counter() - started),
        ) == 27:
            return False
    return True


def _show(
    image: object,
    phase: str,
    trial: Trial,
    index: int,
    total: int,
    remaining: float,
    signal: FaceSignals | None = None,
) -> int:
    import cv2

    display = image.copy()
    cv2.rectangle(display, (0, 0), (display.shape[1], 160), (0, 0, 0), -1)
    lines = (
        f"{phase} | Versuch {index + 1}/{total} | Gruppe {trial.number}/{trial.total} | {max(remaining, 0):.1f}s",
        trial.prompt,
        "Nur eine Geste | Leertaste = Start | Esc = Abbruch | KEINE AUSGABE",
    )
    for row, line in enumerate(lines):
        cv2.putText(display, line, (24, 40 + row * 46), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 255), 2)
    if signal is not None:
        status = f"Linkes/rechtes Auge: {signal.person_left_eye_closed:.2f} / {signal.person_right_eye_closed:.2f}"
        cv2.putText(display, status, (24, display.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
    cv2.imshow(WINDOW_TITLE, display)
    return cv2.waitKey(1) & 0xFF
