from __future__ import annotations

from dataclasses import dataclass

from .config import GestureConfig
from .types import FaceSignals, GestureEvent, GesturePhase


@dataclass(slots=True)
class _EyeState:
    phase: GesturePhase = GesturePhase.IDLE
    candidate_since_ns: int = 0
    cooldown_until_ns: int = 0


class GestureEngine:
    """Timestamp-based gesture state machines; no frame-count assumptions."""

    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config or GestureConfig()
        self._left = _EyeState()
        self._right = _EyeState()

    def update(self, signals: FaceSignals) -> list[GestureEvent]:
        if not signals.face_present or signals.confidence < self.config.min_face_confidence:
            return self.reset(signals.timestamp_ns, "face_or_confidence_lost")

        pose_allowed = (
            abs(signals.yaw_deg) <= self.config.max_abs_yaw_deg
            and abs(signals.pitch_deg) <= self.config.max_abs_pitch_deg
        )
        if not pose_allowed:
            return self.reset(signals.timestamp_ns, "head_pose_blocked")

        events: list[GestureEvent] = []
        events.extend(
            self._update_eye(
                "person_left_eye_close",
                self._left,
                own=signals.person_left_eye_closed,
                other=signals.person_right_eye_closed,
                now_ns=signals.timestamp_ns,
                confidence=signals.confidence,
            )
        )
        events.extend(
            self._update_eye(
                "person_right_eye_close",
                self._right,
                own=signals.person_right_eye_closed,
                other=signals.person_left_eye_closed,
                now_ns=signals.timestamp_ns,
                confidence=signals.confidence,
            )
        )
        return events

    def reset(self, now_ns: int, reason: str) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        for name, state in (
            ("person_left_eye_close", self._left),
            ("person_right_eye_close", self._right),
        ):
            if state.phase == GesturePhase.ACTIVE:
                events.append(GestureEvent(name, False, now_ns, 0.0, reason))
            state.phase = GesturePhase.IDLE
            state.candidate_since_ns = 0
            state.cooldown_until_ns = 0
        return events

    def _update_eye(
        self,
        name: str,
        state: _EyeState,
        *,
        own: float,
        other: float,
        now_ns: int,
        confidence: float,
    ) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        asymmetric = own - other >= self.config.eye_asymmetry_min
        candidate = (
            own >= self.config.eye_close_on
            and other <= self.config.eye_open_guard
            and asymmetric
        )

        if state.phase == GesturePhase.COOLDOWN:
            if now_ns >= state.cooldown_until_ns:
                state.phase = GesturePhase.IDLE
            else:
                return events

        if state.phase == GesturePhase.IDLE and candidate:
            state.phase = GesturePhase.CANDIDATE
            state.candidate_since_ns = now_ns
        elif state.phase == GesturePhase.CANDIDATE:
            if not candidate:
                state.phase = GesturePhase.IDLE
                state.candidate_since_ns = 0
            elif (now_ns - state.candidate_since_ns) / 1_000_000 >= self.config.unilateral_min_ms:
                state.phase = GesturePhase.ACTIVE
                events.append(GestureEvent(name, True, now_ns, confidence, "minimum_duration_met"))
        elif state.phase == GesturePhase.ACTIVE and own <= self.config.eye_close_off:
            events.append(GestureEvent(name, False, now_ns, confidence, "reopen_threshold"))
            state.phase = GesturePhase.COOLDOWN
            state.cooldown_until_ns = now_ns + int(self.config.cooldown_ms * 1_000_000)

        return events
