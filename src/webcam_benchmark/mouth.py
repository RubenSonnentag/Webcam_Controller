from __future__ import annotations

import numpy as np

from .eye_test import _intervals
from .trial_test import _merge_short_gaps
from .types import FaceSignals


MOUTH_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
MOUTH_DURATIONS_MS = (90.0, 120.0, 150.0, 200.0, 250.0, 300.0)
MOUTH_GAPS_MS = (0.0, 35.0, 70.0)


def evaluate_mouth_trials(
    grouped_samples: dict[tuple[str, int], list[FaceSignals]],
) -> dict[str, object]:
    prepared: dict[tuple[str, int], dict[tuple[float, float], float]] = {}
    trial_summaries: list[dict[str, object]] = []
    for key, samples in grouped_samples.items():
        present = [sample for sample in samples if sample.face_present]
        values = np.asarray([sample.jaw_open for sample in present], dtype=float)
        durations: dict[tuple[float, float], float] = {}
        for threshold in MOUTH_THRESHOLDS:
            raw = _intervals(present, lambda sample, limit=threshold: sample.jaw_open >= limit)
            for gap in MOUTH_GAPS_MS:
                merged = _merge_short_gaps(raw, gap)
                durations[(threshold, gap)] = max(
                    ((end - start) / 1e6 for start, end in merged), default=0.0
                )
        prepared[key] = durations
        trial_summaries.append(
            {
                "expected": key[0],
                "number": key[1],
                "frame_count": len(samples),
                "face_present_rate": len(present) / max(len(samples), 1),
                "jaw_open": (
                    {
                        "median": float(np.median(values)),
                        "p95": float(np.percentile(values, 95)),
                        "max": float(np.max(values)),
                    }
                    if values.size
                    else None
                ),
            }
        )

    positives = sum(1 for expected, _ in grouped_samples if expected == "mouth_open")
    negatives = sum(1 for expected, _ in grouped_samples if expected == "mouth_neutral")
    candidates: list[dict[str, object]] = []
    for threshold in MOUTH_THRESHOLDS:
        for gap in MOUTH_GAPS_MS:
            for duration in MOUTH_DURATIONS_MS:
                true_positive = false_positive = 0
                for (expected, _), durations in prepared.items():
                    detected = durations[(threshold, gap)] >= duration
                    true_positive += int(expected == "mouth_open" and detected)
                    false_positive += int(expected == "mouth_neutral" and detected)
                candidates.append(
                    {
                        "parameters": {
                            "jaw_open_threshold": threshold,
                            "minimum_duration_ms": duration,
                            "gap_tolerance_ms": gap,
                        },
                        "true_positive_trials": true_positive,
                        "false_positive_trials": false_positive,
                        "recall": true_positive / max(positives, 1),
                        "false_positive_rate": false_positive / max(negatives, 1),
                    }
                )
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["false_positive_trials"],
            -float(candidate["recall"]),
            float(candidate["parameters"]["minimum_duration_ms"]),  # type: ignore[index]
            float(candidate["parameters"]["gap_tolerance_ms"]),  # type: ignore[index]
            abs(float(candidate["parameters"]["jaw_open_threshold"]) - 0.25),  # type: ignore[index]
        ),
    )
    safe = [candidate for candidate in ranked if candidate["false_positive_trials"] == 0]
    return {
        "objective": "Zero neutral false positives, then mouth-open recall and latency",
        "trial_counts": {"mouth_neutral": negatives, "mouth_open": positives},
        "configuration_count": len(candidates),
        "safe_configuration_count": len(safe),
        "top_safe_candidates": safe[:25],
        "top_overall_candidates": ranked[:25],
        "trials": trial_summaries,
        "warning": "Controlled single-user trials; speaking and natural mouth motion remain independent negatives",
    }
