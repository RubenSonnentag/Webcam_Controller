import pytest

from webcam_benchmark.eye_test import _eye_summary
from webcam_benchmark.types import FaceSignals


def sample(ms: int, left: float, right: float) -> FaceSignals:
    return FaceSignals(ms * 1_000_000, True, person_left_eye_closed=left, person_right_eye_closed=right)


def test_pairs_bilateral_blink_and_rejects_it_as_unilateral() -> None:
    samples = [
        sample(0, 0.1, 0.1),
        sample(33, 0.8, 0.75),
        sample(66, 0.7, 0.72),
        sample(99, 0.1, 0.1),
    ]
    result = _eye_summary(samples)
    assert result["paired_blinks"]["count"] == 1
    assert result["paired_blinks"]["onset_delta_ms"]["max"] == 0.0
    assert result["left_unilateral"]["count"] == 0
    assert result["right_unilateral"]["count"] == 0


def test_detects_sustained_left_unilateral_closure() -> None:
    samples = [sample(0, 0.1, 0.1), sample(33, 0.7, 0.1), sample(66, 0.8, 0.1), sample(99, 0.1, 0.1)]
    result = _eye_summary(samples)
    assert result["left_unilateral"]["count"] == 1
    assert result["left_unilateral"]["duration_ms"]["median"] == pytest.approx(66.0)
