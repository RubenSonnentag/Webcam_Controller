import numpy as np

from webcam_benchmark.backends.openseeface_backend import (
    _eye_aspect_ratio,
    _normalize_angle,
    _pose_degrees,
)


def test_eye_aspect_ratio_uses_six_point_geometry() -> None:
    landmarks = np.zeros((68, 3), dtype=float)
    landmarks[[42, 45], :2] = ((0.0, 0.0), (4.0, 0.0))
    landmarks[[43, 44], :2] = ((1.0, 1.0), (3.0, 1.0))
    landmarks[[47, 46], :2] = ((1.0, -1.0), (3.0, -1.0))
    assert _eye_aspect_ratio(landmarks, (42, 43, 44, 45, 46, 47)) == 0.5


def test_pose_conversion_places_openseeface_neutral_near_zero() -> None:
    yaw, pitch, roll = _pose_degrees((-180.0, 0.0, 90.0))
    assert (yaw, pitch, roll) == (0.0, 0.0, 0.0)


def test_angle_normalization_handles_openseeface_seam() -> None:
    assert _normalize_angle(-359.0) == 1.0
