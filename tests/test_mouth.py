from webcam_benchmark.mouth import evaluate_mouth_trials
from webcam_benchmark.types import FaceSignals


def signal(ms: int, jaw_open: float) -> FaceSignals:
    return FaceSignals(ms * 1_000_000, True, jaw_open=jaw_open)


def test_mouth_sweep_finds_zero_false_positive_candidate() -> None:
    grouped = {
        ("mouth_neutral", 1): [signal(0, 0.01), signal(100, 0.02), signal(200, 0.01)],
        ("mouth_open", 1): [
            signal(0, 0.01),
            signal(50, 0.7),
            signal(200, 0.7),
            signal(250, 0.01),
        ],
    }
    result = evaluate_mouth_trials(grouped)
    best = result["top_safe_candidates"][0]
    assert best["false_positive_trials"] == 0
    assert best["recall"] == 1.0
