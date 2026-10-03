from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GesturePhase(str, Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    arrived_ns: int
    image_bgr: object


@dataclass(frozen=True, slots=True)
class FaceSignals:
    timestamp_ns: int
    face_present: bool
    confidence: float = 0.0
    person_left_eye_closed: float = 0.0
    person_right_eye_closed: float = 0.0
    person_left_eye_aspect_ratio: float = 0.0
    person_right_eye_aspect_ratio: float = 0.0
    face_bbox_normalized: tuple[float, float, float, float] | None = None
    jaw_open: float = 0.0
    brow_inner_up: float = 0.0
    brow_outer_up_left: float = 0.0
    brow_outer_up_right: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    raw: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GestureEvent:
    name: str
    active: bool
    timestamp_ns: int
    confidence: float
    reason: str
