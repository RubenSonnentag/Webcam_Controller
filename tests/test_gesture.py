from webcam_benchmark.config import GestureConfig
from webcam_benchmark.gesture import GestureEngine
from webcam_benchmark.types import FaceSignals


def signals(ms: float, left: float, right: float, *, yaw: float = 0.0) -> FaceSignals:
    return FaceSignals(
        timestamp_ns=int(ms * 1_000_000),
        face_present=True,
        confidence=1.0,
        person_left_eye_closed=left,
        person_right_eye_closed=right,
        yaw_deg=yaw,
    )


def test_short_unilateral_closure_is_not_triggered() -> None:
    engine = GestureEngine(GestureConfig(unilateral_min_ms=100))
    assert engine.update(signals(0, 0.9, 0.05)) == []
    assert engine.update(signals(50, 0.9, 0.05)) == []
    assert engine.update(signals(80, 0.05, 0.05)) == []


def test_sustained_left_closure_triggers_and_releases() -> None:
    engine = GestureEngine(GestureConfig(unilateral_min_ms=100))
    engine.update(signals(0, 0.9, 0.05))
    events = engine.update(signals(101, 0.9, 0.05))
    assert [(event.name, event.active) for event in events] == [("person_left_eye_close", True)]
    events = engine.update(signals(150, 0.05, 0.05))
    assert [(event.name, event.active) for event in events] == [("person_left_eye_close", False)]


def test_bilateral_blink_does_not_trigger_unilateral_action() -> None:
    engine = GestureEngine(GestureConfig(unilateral_min_ms=50))
    assert engine.update(signals(0, 0.9, 0.88)) == []
    assert engine.update(signals(70, 0.9, 0.88)) == []


def test_head_rotation_blocks_eye_action() -> None:
    engine = GestureEngine(GestureConfig(unilateral_min_ms=50, max_abs_yaw_deg=25))
    engine.update(signals(0, 0.9, 0.05, yaw=30))
    assert engine.update(signals(100, 0.9, 0.05, yaw=30)) == []
