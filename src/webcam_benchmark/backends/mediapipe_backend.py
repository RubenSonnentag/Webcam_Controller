from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from ..types import FaceSignals
from .base import FaceBackend


class MediaPipeFaceBackend(FaceBackend):
    def __init__(self, model_path: Path) -> None:
        cache_dir = Path.cwd() / ".cache" / "matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import mediapipe as mp

        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def analyze(self, image_bgr: object, timestamp_ns: int) -> FaceSignals:
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(timestamp_ns // 1_000_000, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return FaceSignals(timestamp_ns=timestamp_ns, face_present=False)

        shapes = {
            item.category_name: float(item.score)
            for item in result.face_blendshapes[0]
        } if result.face_blendshapes else {}
        landmarks = result.face_landmarks[0]
        xs = [float(landmark.x) for landmark in landmarks]
        ys = [float(landmark.y) for landmark in landmarks]
        face_bbox = (min(xs), min(ys), max(xs), max(ys))
        # MediaPipe's LEFT/RIGHT naming is anatomical (the person's side).
        left_ear = _eye_aspect_ratio(landmarks, (362, 385, 387, 263, 373, 380))
        right_ear = _eye_aspect_ratio(landmarks, (33, 160, 158, 133, 153, 144))
        yaw, pitch, roll = _pose_degrees(result.facial_transformation_matrixes)
        confidence = _signal_confidence(shapes)
        return FaceSignals(
            timestamp_ns=timestamp_ns,
            face_present=True,
            confidence=confidence,
            person_left_eye_closed=shapes.get("eyeBlinkLeft", 0.0),
            person_right_eye_closed=shapes.get("eyeBlinkRight", 0.0),
            person_left_eye_aspect_ratio=left_ear,
            person_right_eye_aspect_ratio=right_ear,
            face_bbox_normalized=face_bbox,
            jaw_open=shapes.get("jawOpen", 0.0),
            brow_inner_up=shapes.get("browInnerUp", 0.0),
            brow_outer_up_left=shapes.get("browOuterUpLeft", 0.0),
            brow_outer_up_right=shapes.get("browOuterUpRight", 0.0),
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            raw=shapes,
        )

    def close(self) -> None:
        self._landmarker.close()


def _eye_aspect_ratio(landmarks: object, indices: tuple[int, int, int, int, int, int]) -> float:
    points = [landmarks[index] for index in indices]  # type: ignore[index]

    def distance(a: object, b: object) -> float:
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))  # type: ignore[attr-defined]

    horizontal = distance(points[0], points[3])
    if horizontal <= 1e-9:
        return 0.0
    return (distance(points[1], points[5]) + distance(points[2], points[4])) / (2.0 * horizontal)


def _signal_confidence(shapes: dict[str, float]) -> float:
    # Face Landmarker does not expose a single face-presence score in the result.
    # Presence has already passed MediaPipe's configured threshold; keep this value
    # explicit so it is not mistaken for a calibrated probability.
    return 1.0 if shapes else 0.5


def _pose_degrees(matrices: object) -> tuple[float, float, float]:
    if not matrices:
        return 0.0, 0.0, 0.0
    matrix = np.asarray(matrices[0], dtype=np.float64)
    rotation = matrix[:3, :3]
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = 0.0
    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )
