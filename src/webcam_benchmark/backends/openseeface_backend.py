from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from ..types import FaceSignals
from .base import FaceBackend


class OpenSeeFaceBackend(FaceBackend):
    """Adapter for an unmodified local OpenSeeFace checkout.

    OpenSeeFace remains an optional external dependency. Its source and models
    are not copied into this project; ``root`` must point at the official
    checkout containing tracker.py and models/.
    """

    def __init__(
        self,
        root: Path,
        *,
        model_type: int = 4,
        max_threads: int = 4,
        known_face_roi: bool = True,
    ) -> None:
        root = root.resolve()
        tracker_path = root / "tracker.py"
        model_dir = root / "models"
        license_path = root / "LICENSE"
        if not tracker_path.is_file() or not model_dir.is_dir():
            raise FileNotFoundError(
                "OpenSeeFace root must contain tracker.py and models/: " f"{root}"
            )
        if not license_path.is_file():
            raise FileNotFoundError(f"OpenSeeFace LICENSE is missing: {license_path}")
        if model_type not in (-3, -2, -1, 0, 1, 2, 3, 4):
            raise ValueError("Unsupported OpenSeeFace model type")
        if max_threads < 1:
            raise ValueError("max_threads must be positive")

        self._root = root
        self._model_dir = model_dir
        self._model_type = model_type
        self._max_threads = max_threads
        self._known_face_roi = known_face_roi
        self._module = _load_tracker_module(root, tracker_path)
        self._tracker: object | None = None
        self._frame_size: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return f"OpenSeeFace-model-{self._model_type}"

    def analyze(self, image_bgr: object, timestamp_ns: int) -> FaceSignals:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("OpenSeeFace requires a BGR color image")
        height, width = image.shape[:2]
        self._ensure_tracker(width, height)
        additional_faces = [(0, 0, width, height)] if self._known_face_roi else []
        faces = self._tracker.predict(image, additional_faces)  # type: ignore[union-attr]
        if not faces:
            return FaceSignals(timestamp_ns=timestamp_ns, face_present=False)

        face = faces[0]
        features = dict(face.current_features or {})
        left_open = float(face.eye_blink[1]) if face.eye_blink is not None else 1.0
        right_open = float(face.eye_blink[0]) if face.eye_blink is not None else 1.0
        landmarks = np.asarray(face.lms, dtype=float)
        left_ear = _eye_aspect_ratio(landmarks, (42, 43, 44, 45, 46, 47))
        right_ear = _eye_aspect_ratio(landmarks, (36, 37, 38, 39, 40, 41))
        yaw, pitch, roll = _pose_degrees(face.euler if face.success else None)
        bbox = _normalized_bbox(landmarks, width, height)
        brow_left = float(features.get("eyebrow_updown_l", 0.0))
        brow_right = float(features.get("eyebrow_updown_r", 0.0))
        return FaceSignals(
            timestamp_ns=timestamp_ns,
            face_present=True,
            confidence=float(face.conf),
            person_left_eye_closed=float(np.clip(1.0 - left_open, 0.0, 1.0)),
            person_right_eye_closed=float(np.clip(1.0 - right_open, 0.0, 1.0)),
            person_left_eye_aspect_ratio=left_ear,
            person_right_eye_aspect_ratio=right_ear,
            face_bbox_normalized=bbox,
            jaw_open=float(features.get("mouth_open", 0.0)),
            brow_inner_up=(brow_left + brow_right) / 2.0,
            brow_outer_up_left=brow_left,
            brow_outer_up_right=brow_right,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            raw={key: float(value) for key, value in features.items()},
        )

    def _ensure_tracker(self, width: int, height: int) -> None:
        if self._tracker is not None:
            if self._frame_size != (width, height):
                raise ValueError("OpenSeeFace frame size changed during one session")
            return
        tracker_type = self._module.Tracker
        self._tracker = tracker_type(
            width,
            height,
            model_type=self._model_type,
            model_dir=str(self._model_dir),
            max_faces=1,
            max_threads=self._max_threads,
            silent=True,
            no_gaze=True,
            max_feature_updates=900,
        )
        self._frame_size = (width, height)


def _load_tracker_module(root: Path, tracker_path: Path) -> ModuleType:
    try:
        import onnxruntime  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "OpenSeeFace requires the optional dependency: "
            "pip install -e .[openseeface]"
        ) from error
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_name = "_webcam_controller_openseeface_tracker"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, tracker_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load OpenSeeFace tracker: {tracker_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _eye_aspect_ratio(landmarks: np.ndarray, indices: tuple[int, ...]) -> float:
    points = landmarks[list(indices), :2]
    horizontal = float(np.linalg.norm(points[0] - points[3]))
    if horizontal <= 1e-9:
        return 0.0
    vertical = float(np.linalg.norm(points[1] - points[5]))
    vertical += float(np.linalg.norm(points[2] - points[4]))
    return vertical / (2.0 * horizontal)


def _normalized_bbox(
    landmarks: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float]:
    points = landmarks[:66, :2]
    return (
        float(np.min(points[:, 0]) / width),
        float(np.min(points[:, 1]) / height),
        float(np.max(points[:, 0]) / width),
        float(np.max(points[:, 1]) / height),
    )


def _pose_degrees(euler: object | None) -> tuple[float, float, float]:
    if euler is None:
        return 0.0, 0.0, 0.0
    raw_x, raw_y, raw_z = (float(value) for value in euler)  # type: ignore[arg-type]
    # Equivalent to the coordinate conversion in OpenSeeFace's OpenSee.cs,
    # normalized so neutral pose remains close to zero across the ±180 seam.
    pitch = _normalize_angle(-(raw_x + 180.0))
    yaw = _normalize_angle(-raw_y)
    roll = _normalize_angle(raw_z - 90.0)
    return yaw, pitch, roll


def _normalize_angle(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0
