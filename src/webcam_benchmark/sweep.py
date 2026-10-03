from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trial_test import _merge_short_gaps, _numeric_intervals
from .types import FaceSignals


@dataclass(frozen=True, slots=True)
class SweepRanges:
    close_thresholds: tuple[float, ...] = (0.35, 0.40, 0.425, 0.45, 0.50)
    left_asymmetries: tuple[float, ...] = (0.18, 0.22, 0.26, 0.30)
    right_asymmetries: tuple[float, ...] = (0.18, 0.22, 0.26, 0.30)
    durations_ms: tuple[float, ...] = (60.0, 75.0, 90.0, 100.0, 120.0, 150.0)
    smoothing_ms: tuple[float, ...] = (0.0, 35.0, 70.0)
    gap_tolerance_ms: tuple[float, ...] = (0.0, 35.0, 70.0)


def run_parameter_sweep(
    grouped_samples: dict[tuple[str, int], list[FaceSignals]],
    ranges: SweepRanges | None = None,
    production_duration_floor_ms: float = 90.0,
) -> dict[str, object]:
    ranges = ranges or SweepRanges()
    prepared = {
        key: _prepare_closedness(samples, ranges.smoothing_ms)
        for key, samples in grouped_samples.items()
    }
    totals = {
        label: sum(1 for expected, _ in grouped_samples if expected == label)
        for label in ("normal_blink", "left_eye", "right_eye")
    }
    candidates: list[dict[str, object]] = []
    for close in ranges.close_thresholds:
        for left_asymmetry in ranges.left_asymmetries:
            for right_asymmetry in ranges.right_asymmetries:
                for smoothing in ranges.smoothing_ms:
                    for gap in ranges.gap_tolerance_ms:
                        durations_by_trial = {
                            key: _candidate_durations(
                                values, close, left_asymmetry, right_asymmetry, smoothing, gap
                            )
                            for key, values in prepared.items()
                        }
                        for duration in ranges.durations_ms:
                            candidates.append(
                                _score_configuration(
                                    durations_by_trial,
                                    totals,
                                    close,
                                    left_asymmetry,
                                    right_asymmetry,
                                    duration,
                                    smoothing,
                                    gap,
                                )
                            )
    ranked = sorted(candidates, key=_rank_key)
    observed_safe = [
        candidate
        for candidate in ranked
        if candidate["false_positive_blinks"] == 0
        and candidate["side_confusions"] == 0
        and candidate["ambiguous"] == 0
    ]
    eligible_safe = [
        candidate
        for candidate in observed_safe
        if float(candidate["parameters"]["minimum_duration_ms"])  # type: ignore[index]
        >= production_duration_floor_ms
    ]
    return {
        "objective": "Safety first, then minimum-side recall, total recall, duration and filtering cost",
        "dataset_trials": totals,
        "configuration_count": len(candidates),
        "observed_safe_configuration_count": len(observed_safe),
        "production_duration_floor_ms": production_duration_floor_ms,
        "eligible_safe_configuration_count": len(eligible_safe),
        "top_safe_candidates": eligible_safe[:25],
        "top_exploratory_below_floor": [
            candidate
            for candidate in observed_safe
            if float(candidate["parameters"]["minimum_duration_ms"])  # type: ignore[index]
            < production_duration_floor_ms
        ][:10],
        "top_overall_candidates": ranked[:25],
        "warning": "Exploratory single-user dataset; validate selected settings on independent conditions",
    }


def _prepare_closedness(
    samples: list[FaceSignals], smoothing_options: tuple[float, ...]
) -> dict[float, tuple[list[FaceSignals], np.ndarray, np.ndarray]]:
    if not samples:
        return {option: ([], np.asarray([]), np.asarray([])) for option in smoothing_options}
    left_ear = np.asarray([sample.person_left_eye_aspect_ratio for sample in samples], dtype=float)
    right_ear = np.asarray([sample.person_right_eye_aspect_ratio for sample in samples], dtype=float)
    left_open = float(np.percentile(left_ear, 90))
    right_open = float(np.percentile(right_ear, 90))
    left = 1.0 - left_ear / max(left_open, 1e-9)
    right = 1.0 - right_ear / max(right_open, 1e-9)
    timestamps = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    return {
        option: (
            samples,
            _causal_time_average(left, timestamps, option),
            _causal_time_average(right, timestamps, option),
        )
        for option in smoothing_options
    }


def _causal_time_average(
    values: np.ndarray, timestamps_ns: np.ndarray, window_ms: float
) -> np.ndarray:
    if window_ms <= 0 or values.size == 0:
        return values.copy()
    output = np.empty_like(values)
    window_ns = int(window_ms * 1e6)
    start = 0
    running = 0.0
    for index, value in enumerate(values):
        running += float(value)
        while timestamps_ns[index] - timestamps_ns[start] > window_ns:
            running -= float(values[start])
            start += 1
        output[index] = running / (index - start + 1)
    return output


def _candidate_durations(
    prepared: dict[float, tuple[list[FaceSignals], np.ndarray, np.ndarray]],
    close: float,
    left_asymmetry: float,
    right_asymmetry: float,
    smoothing_ms: float,
    gap_ms: float,
) -> tuple[float, float]:
    left_intervals, right_intervals = _candidate_intervals(
        prepared, close, left_asymmetry, right_asymmetry, smoothing_ms, gap_ms
    )
    left_max = max(((end - start) / 1e6 for start, end in left_intervals), default=0.0)
    right_max = max(((end - start) / 1e6 for start, end in right_intervals), default=0.0)
    return left_max, right_max


def _candidate_intervals(
    prepared: dict[float, tuple[list[FaceSignals], np.ndarray, np.ndarray]],
    close: float,
    left_asymmetry: float,
    right_asymmetry: float,
    smoothing_ms: float,
    gap_ms: float,
    pose_allowed: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    samples, left, right = prepared[smoothing_ms]
    if not samples:
        return [], []
    if pose_allowed is not None:
        left = np.where(pose_allowed, left, -np.inf)
        right = np.where(pose_allowed, right, -np.inf)
    indexed = list(zip(samples, left, right))
    left_intervals = _numeric_intervals(
        indexed, lambda own, other: own >= close and own - other >= left_asymmetry
    )
    right_intervals = _numeric_intervals(
        indexed, lambda other, own: own >= close and own - other >= right_asymmetry
    )
    left_intervals = _merge_short_gaps(left_intervals, gap_ms)
    right_intervals = _merge_short_gaps(right_intervals, gap_ms)
    return left_intervals, right_intervals


def evaluate_negative_stream(
    samples: list[FaceSignals],
    *,
    close: float = 0.45,
    left_asymmetry: float = 0.22,
    right_asymmetry: float = 0.18,
    duration_ms: float = 90.0,
    smoothing_ms: float = 0.0,
    gap_ms: float = 70.0,
    neutral_yaw_deg: float | None = None,
    neutral_pitch_deg: float | None = None,
    max_yaw_delta_deg: float = 15.0,
    max_pitch_delta_deg: float = 18.0,
    yaw_reenable_delta_deg: float | None = None,
    pitch_reenable_delta_deg: float | None = None,
    pose_reenable_ms: float = 150.0,
) -> dict[str, object]:
    prepared = _prepare_closedness(samples, (smoothing_ms,))
    pose_allowed = None
    if neutral_yaw_deg is not None and neutral_pitch_deg is not None:
        pose_allowed = _pose_gate_mask(
            samples,
            neutral_yaw_deg,
            neutral_pitch_deg,
            max_yaw_delta_deg,
            max_pitch_delta_deg,
            yaw_reenable_delta_deg,
            pitch_reenable_delta_deg,
            pose_reenable_ms,
        )
    left, right = _candidate_intervals(
        prepared,
        close,
        left_asymmetry,
        right_asymmetry,
        smoothing_ms,
        gap_ms,
        pose_allowed,
    )
    qualified_left = [(start, end) for start, end in left if (end - start) / 1e6 >= duration_ms]
    qualified_right = [(start, end) for start, end in right if (end - start) / 1e6 >= duration_ms]
    span_s = (
        (samples[-1].timestamp_ns - samples[0].timestamp_ns) / 1e9
        if len(samples) > 1
        else 0.0
    )
    total = len(qualified_left) + len(qualified_right)
    return {
        "parameters": {
            "close_threshold": close,
            "left_asymmetry": left_asymmetry,
            "right_asymmetry": right_asymmetry,
            "minimum_duration_ms": duration_ms,
            "causal_smoothing_ms": smoothing_ms,
            "gap_tolerance_ms": gap_ms,
            "pose_yaw_block_deg": max_yaw_delta_deg,
            "pose_pitch_block_deg": max_pitch_delta_deg,
            "pose_yaw_reenable_deg": (
                yaw_reenable_delta_deg
                if yaw_reenable_delta_deg is not None
                else max(max_yaw_delta_deg - 3.0, 0.0)
            ),
            "pose_pitch_reenable_deg": (
                pitch_reenable_delta_deg
                if pitch_reenable_delta_deg is not None
                else max(max_pitch_delta_deg - 3.0, 0.0)
            ),
            "pose_reenable_ms": pose_reenable_ms,
        },
        "duration_s": span_s,
        "left_false_events": len(qualified_left),
        "right_false_events": len(qualified_right),
        "total_false_events": total,
        "false_events_per_minute": total * 60.0 / span_s if span_s > 0 else 0.0,
        "pose_gate_enabled": pose_allowed is not None,
        "pose_blocked_frame_rate": (
            1.0 - float(np.mean(pose_allowed)) if pose_allowed is not None and pose_allowed.size else 0.0
        ),
    }


def _pose_gate_mask(
    samples: list[FaceSignals],
    neutral_yaw_deg: float,
    neutral_pitch_deg: float,
    yaw_block_deg: float,
    pitch_block_deg: float,
    yaw_reenable_deg: float | None,
    pitch_reenable_deg: float | None,
    reenable_ms: float,
) -> np.ndarray:
    yaw_reenable = (
        yaw_reenable_deg
        if yaw_reenable_deg is not None
        else max(yaw_block_deg - 3.0, 0.0)
    )
    pitch_reenable = (
        pitch_reenable_deg
        if pitch_reenable_deg is not None
        else max(pitch_block_deg - 3.0, 0.0)
    )
    allowed = True
    stable_since_ns: int | None = None
    result: list[bool] = []
    for sample in samples:
        yaw_delta = abs(sample.yaw_deg - neutral_yaw_deg)
        pitch_delta = abs(sample.pitch_deg - neutral_pitch_deg)
        if not sample.face_present:
            allowed = False
            stable_since_ns = None
        elif allowed and (yaw_delta > yaw_block_deg or pitch_delta > pitch_block_deg):
            allowed = False
            stable_since_ns = None
        elif not allowed:
            if yaw_delta <= yaw_reenable and pitch_delta <= pitch_reenable:
                if stable_since_ns is None:
                    stable_since_ns = sample.timestamp_ns
                if (sample.timestamp_ns - stable_since_ns) / 1e6 >= reenable_ms:
                    allowed = True
                    stable_since_ns = None
            else:
                stable_since_ns = None
        result.append(allowed)
    return np.asarray(result, dtype=bool)


def _score_configuration(
    durations_by_trial: dict[tuple[str, int], tuple[float, float]],
    totals: dict[str, int],
    close: float,
    left_asymmetry: float,
    right_asymmetry: float,
    duration: float,
    smoothing: float,
    gap: float,
) -> dict[str, object]:
    correct_left = correct_right = false_blinks = side_confusions = ambiguous = 0
    for (expected, _), (left_ms, right_ms) in durations_by_trial.items():
        left = left_ms >= duration
        right = right_ms >= duration
        if left and right:
            detected = "ambiguous"
        elif left:
            detected = "left_eye"
        elif right:
            detected = "right_eye"
        else:
            detected = "none"
        if expected == "normal_blink" and detected != "none":
            false_blinks += 1
        elif expected == "left_eye":
            correct_left += int(detected == "left_eye")
            side_confusions += int(detected == "right_eye")
            ambiguous += int(detected == "ambiguous")
        elif expected == "right_eye":
            correct_right += int(detected == "right_eye")
            side_confusions += int(detected == "left_eye")
            ambiguous += int(detected == "ambiguous")
    left_total = max(totals.get("left_eye", 0), 1)
    right_total = max(totals.get("right_eye", 0), 1)
    total_unilateral = max(totals.get("left_eye", 0) + totals.get("right_eye", 0), 1)
    return {
        "parameters": {
            "close_threshold": close,
            "left_asymmetry": left_asymmetry,
            "right_asymmetry": right_asymmetry,
            "minimum_duration_ms": duration,
            "causal_smoothing_ms": smoothing,
            "gap_tolerance_ms": gap,
        },
        "correct_left": correct_left,
        "correct_right": correct_right,
        "unilateral_recall": (correct_left + correct_right) / total_unilateral,
        "minimum_side_recall": min(correct_left / left_total, correct_right / right_total),
        "false_positive_blinks": false_blinks,
        "side_confusions": side_confusions,
        "ambiguous": ambiguous,
    }


def _rank_key(candidate: dict[str, object]) -> tuple[object, ...]:
    parameters = candidate["parameters"]  # type: ignore[assignment]
    return (
        candidate["false_positive_blinks"],
        candidate["side_confusions"],
        candidate["ambiguous"],
        -float(candidate["minimum_side_recall"]),
        -float(candidate["unilateral_recall"]),
        float(parameters["minimum_duration_ms"]),  # type: ignore[index]
        float(parameters["causal_smoothing_ms"]),  # type: ignore[index]
        float(parameters["gap_tolerance_ms"]),  # type: ignore[index]
        abs(float(parameters["close_threshold"]) - 0.45),  # type: ignore[index]
        abs(float(parameters["left_asymmetry"]) - 0.23),  # type: ignore[index]
        abs(float(parameters["right_asymmetry"]) - 0.22),  # type: ignore[index]
    )
