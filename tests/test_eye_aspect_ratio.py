from dataclasses import dataclass

import pytest

from webcam_benchmark.backends.mediapipe_backend import _eye_aspect_ratio


@dataclass
class Point:
    x: float
    y: float


def test_eye_aspect_ratio_uses_vertical_opening_over_width() -> None:
    points = [
        Point(0.0, 0.0),
        Point(0.25, 0.2),
        Point(0.75, 0.2),
        Point(1.0, 0.0),
        Point(0.75, -0.2),
        Point(0.25, -0.2),
    ]
    assert _eye_aspect_ratio(points, (0, 1, 2, 3, 4, 5)) == pytest.approx(0.4)


def test_eye_aspect_ratio_handles_zero_width() -> None:
    points = [Point(0.0, 0.0) for _ in range(6)]
    assert _eye_aspect_ratio(points, (0, 1, 2, 3, 4, 5)) == 0.0
