import pytest

from webcam_benchmark.guided import _summarize
from webcam_benchmark.types import FaceSignals


def test_guided_summary_contains_only_aggregate_values() -> None:
    samples = [
        FaceSignals(1, True, person_left_eye_closed=0.1, person_right_eye_closed=0.2),
        FaceSignals(2, True, person_left_eye_closed=0.9, person_right_eye_closed=0.3),
        FaceSignals(3, False),
    ]

    summary = _summarize(samples)

    assert summary["frames"] == 3
    assert summary["face_present_rate"] == 2 / 3
    assert summary["person_left_eye_closed"]["min"] == 0.1
    assert summary["person_left_eye_closed"]["max"] == 0.9
    assert summary["left_minus_right_eye"]["max"] == pytest.approx(0.6)
