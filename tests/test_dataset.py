import numpy as np

from webcam_benchmark.dataset import _crop_face_roi, _pose_threshold_sweep
from webcam_benchmark.types import FaceSignals


def test_face_roi_has_fixed_size() -> None:
    image = np.full((100, 200, 3), 127, dtype=np.uint8)
    roi = _crop_face_roi(image, (0.25, 0.2, 0.75, 0.8), 64)
    assert roi.shape == (64, 64, 3)
    assert int(roi.mean()) == 127


def test_missing_face_produces_black_privacy_frame() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    roi = _crop_face_roi(image, None, 32)
    assert roi.shape == (32, 32, 3)
    assert not np.any(roi)


def test_pose_threshold_sweep_uses_relative_neutral_pose() -> None:
    samples = [
        FaceSignals(0, True, yaw_deg=5.0, pitch_deg=10.0),
        FaceSignals(1, True, yaw_deg=20.0, pitch_deg=-5.0),
        FaceSignals(2, True, yaw_deg=35.0, pitch_deg=40.0),
    ]
    result = _pose_threshold_sweep(samples, neutral_yaw=5.0, neutral_pitch=10.0)
    assert result["yaw_blocked_rate"]["12"] == 2 / 3
    assert result["yaw_blocked_rate"]["30"] == 0.0
    assert result["pitch_blocked_rate"]["12"] == 2 / 3
