from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GestureConfig:
    eye_close_on: float = 0.62
    eye_close_off: float = 0.42
    eye_open_guard: float = 0.42
    eye_asymmetry_min: float = 0.28
    unilateral_min_ms: float = 100.0
    blink_pair_max_delta_ms: float = 75.0
    cooldown_ms: float = 250.0
    max_abs_yaw_deg: float = 28.0
    max_abs_pitch_deg: float = 22.0
    min_face_confidence: float = 0.5


@dataclass(slots=True)
class CameraConfig:
    index: int = 0
    width: int = 1920
    height: int = 1080
    fps: float = 60.0
    analysis_rate: float = 60.0
    preview: bool = True
    backend: str = "auto"
    fourcc: str = "auto"
