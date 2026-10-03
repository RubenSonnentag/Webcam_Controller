from webcam_benchmark.trial_test import Trial, _classify_trial, _merge_short_gaps, _summarize_trials, _trials
from webcam_benchmark.types import FaceSignals


def signal(ms: int, left: float, right: float, left_ear: float = 0.4, right_ear: float = 0.4) -> FaceSignals:
    return FaceSignals(
        ms * 1_000_000,
        True,
        person_left_eye_closed=left,
        person_right_eye_closed=right,
        person_left_eye_aspect_ratio=left_ear,
        person_right_eye_aspect_ratio=right_ear,
    )


def test_classifies_one_left_eye_trial_at_all_durations() -> None:
    samples = [signal(0, 0.1, 0.1), signal(50, 0.8, 0.1), signal(200, 0.8, 0.1), signal(250, 0.1, 0.1)]
    result = _classify_trial(Trial("left_eye", "", 1), samples)
    assert result["decisions"] == {
        "90": "left_eye",
        "100": "left_eye",
        "120": "left_eye",
        "150": "left_eye",
    }


def test_classifies_paired_blink_without_unilateral_action() -> None:
    samples = [signal(0, 0.1, 0.1), signal(50, 0.8, 0.75), signal(180, 0.7, 0.72), signal(220, 0.1, 0.1)]
    result = _classify_trial(Trial("normal_blink", "", 1), samples)
    assert result["decisions"]["100"] == "normal_blink"


def test_summary_builds_confusion_counts() -> None:
    results = [
        {"expected": "left_eye", "decisions": {"90": "left_eye", "100": "left_eye", "120": "left_eye", "150": "none"}},
        {"expected": "left_eye", "decisions": {"90": "right_eye", "100": "right_eye", "120": "none", "150": "none"}},
    ]
    summary = _summarize_trials(results)
    assert summary["100"]["left_eye"] == {"left_eye": 1, "right_eye": 1}
    assert summary["150"]["left_eye"] == {"none": 2}


def test_short_signal_gap_is_merged_without_adding_release_time() -> None:
    intervals = [(0, 80_000_000), (130_000_000, 220_000_000), (400_000_000, 450_000_000)]
    assert _merge_short_gaps(intervals, 70.0) == [
        (0, 220_000_000),
        (400_000_000, 450_000_000),
    ]


def test_trial_count_is_configurable() -> None:
    trials = _trials(5)
    assert len(trials) == 15
    assert all(trial.total == 5 for trial in trials)


def test_trial_groups_can_be_restricted() -> None:
    trials = _trials(5, ("normal_blink", "right_eye"))
    assert len(trials) == 10
    assert {trial.expected for trial in trials} == {"normal_blink", "right_eye"}


def test_ear_channel_classifies_left_closure() -> None:
    samples = [
        signal(0, 0.1, 0.1),
        signal(50, 0.1, 0.1, 0.12, 0.4),
        signal(200, 0.1, 0.1, 0.12, 0.4),
        signal(250, 0.1, 0.1),
    ]
    result = _classify_trial(Trial("left_eye", "", 1), samples)
    assert result["ear_decisions"] == {
        "90": "left_eye",
        "100": "left_eye",
        "120": "left_eye",
        "150": "left_eye",
    }
