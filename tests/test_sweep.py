import numpy as np

from webcam_benchmark.sweep import (
    SweepRanges,
    _causal_time_average,
    evaluate_negative_stream,
    run_parameter_sweep,
)
from webcam_benchmark.types import FaceSignals


def eye_signal(
    ms: int,
    left_ear: float,
    right_ear: float,
    *,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> FaceSignals:
    return FaceSignals(
        timestamp_ns=ms * 1_000_000,
        face_present=True,
        person_left_eye_aspect_ratio=left_ear,
        person_right_eye_aspect_ratio=right_ear,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
    )


def sequence(closed: str) -> list[FaceSignals]:
    left = 0.1 if closed in ("left", "both") else 0.4
    right = 0.1 if closed in ("right", "both") else 0.4
    return [
        eye_signal(0, 0.4, 0.4),
        eye_signal(50, left, right),
        eye_signal(150, left, right),
        eye_signal(200, 0.4, 0.4),
    ]


def test_parameter_sweep_finds_safe_perfect_configuration() -> None:
    grouped = {
        ("normal_blink", 1): sequence("both"),
        ("left_eye", 1): sequence("left"),
        ("right_eye", 1): sequence("right"),
    }
    ranges = SweepRanges(
        close_thresholds=(0.4,),
        left_asymmetries=(0.2,),
        right_asymmetries=(0.2,),
        durations_ms=(90.0,),
        smoothing_ms=(0.0,),
        gap_tolerance_ms=(0.0,),
    )
    result = run_parameter_sweep(grouped, ranges)
    best = result["top_safe_candidates"][0]
    assert result["configuration_count"] == 1
    assert result["eligible_safe_configuration_count"] == 1
    assert best["unilateral_recall"] == 1.0
    assert best["false_positive_blinks"] == 0


def test_causal_average_does_not_use_future_values() -> None:
    values = np.asarray([0.0, 1.0, 1.0])
    timestamps = np.asarray([0, 30_000_000, 60_000_000])
    smoothed = _causal_time_average(values, timestamps, 35.0)
    assert list(smoothed) == [0.0, 0.5, 1.0]


def test_negative_stream_ignores_bilateral_blink() -> None:
    samples = [
        eye_signal(0, 0.4, 0.4),
        eye_signal(50, 0.1, 0.1),
        eye_signal(150, 0.1, 0.1),
        eye_signal(200, 0.4, 0.4),
    ]
    result = evaluate_negative_stream(samples)
    assert result["total_false_events"] == 0


def test_negative_stream_counts_sustained_unilateral_event() -> None:
    samples = [
        eye_signal(0, 0.4, 0.4),
        eye_signal(50, 0.1, 0.4),
        eye_signal(150, 0.1, 0.4),
        eye_signal(200, 0.4, 0.4),
    ]
    result = evaluate_negative_stream(samples)
    assert result["left_false_events"] == 1
    assert result["right_false_events"] == 0


def test_relative_pose_gate_blocks_unilateral_event_during_head_rotation() -> None:
    samples = [
        eye_signal(0, 0.4, 0.4, yaw_deg=30.0),
        eye_signal(50, 0.1, 0.4, yaw_deg=30.0),
        eye_signal(150, 0.1, 0.4, yaw_deg=30.0),
        eye_signal(200, 0.4, 0.4, yaw_deg=30.0),
    ]
    ungated = evaluate_negative_stream(samples)
    gated = evaluate_negative_stream(
        samples, neutral_yaw_deg=0.0, neutral_pitch_deg=0.0
    )
    assert ungated["left_false_events"] == 1
    assert gated["total_false_events"] == 0
    assert gated["pose_blocked_frame_rate"] == 1.0


def test_pose_gate_requires_stable_return_before_reenabling() -> None:
    samples = [
        eye_signal(0, 0.4, 0.4),
        eye_signal(50, 0.4, 0.4, yaw_deg=20.0),
        eye_signal(100, 0.1, 0.4),
        eye_signal(200, 0.1, 0.4),
        eye_signal(260, 0.4, 0.4),
    ]
    gated = evaluate_negative_stream(
        samples, neutral_yaw_deg=0.0, neutral_pitch_deg=0.0
    )
    assert gated["total_false_events"] == 0
    assert gated["pose_blocked_frame_rate"] == 0.6
