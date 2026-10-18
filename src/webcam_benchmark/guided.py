from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .backends.base import FaceBackend
from .capture import CameraCapture
from .config import CameraConfig
from .latest_frame import LatestFrameBuffer
from .types import FaceSignals


@dataclass(frozen=True, slots=True)
class ProtocolStep:
    name: str
    prompt: str
    duration_s: float


PROTOCOL = (
    ProtocolStep("neutral", "Entspannt geradeaus schauen", 5.0),
    ProtocolStep("normal_blinks", "3-5 Mal normal blinzeln", 7.0),
    ProtocolStep("left_eye_close", "LINKES Auge der Person schliessen und halten", 5.0),
    ProtocolStep("right_eye_close", "RECHTES Auge der Person schliessen und halten", 5.0),
    ProtocolStep("mouth_open", "Mund deutlich oeffnen und halten", 5.0),
    ProtocolStep("brows_up", "Beide Augenbrauen hochziehen und halten", 5.0),
    ProtocolStep("head_left", "Kopf nach LINKS drehen und halten", 4.0),
    ProtocolStep("head_right", "Kopf nach RECHTS drehen und halten", 4.0),
    ProtocolStep("head_up", "Kopf nach OBEN neigen und halten", 4.0),
    ProtocolStep("head_down", "Kopf nach UNTEN neigen und halten", 4.0),
)

WINDOW_TITLE = "Gefuehrter Phase-2-Gestentest (keine Ausgabe)"


def run_guided(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    prepare_s: float = 3.0,
    export_path: Path | None = None,
) -> dict[str, object]:
    """Run a prompted, output-free protocol and retain only aggregate signals."""
    import cv2

    if not camera_config.preview:
        raise ValueError("The guided protocol requires the preview window")

    buffer = LatestFrameBuffer()
    capture = CameraCapture(camera_config, buffer)
    samples: dict[str, list[FaceSignals]] = {step.name: [] for step in PROTOCOL}
    aborted = False
    capture.start()
    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1280, 720)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
        for step_index, step in enumerate(PROTOCOL):
            while True:
                frame = buffer.take(0.25)
                if capture.error:
                    raise capture.error
                if frame is None:
                    continue
                key = _show_frame(
                    frame.image_bgr,
                    "BEREIT? LEERTASTE DRUECKEN",
                    step.prompt,
                    0.0,
                    step_index,
                )
                if key == 27:
                    aborted = True
                    break
                if key == 32:
                    break
            if aborted:
                break

            phase_start = time.perf_counter()
            while time.perf_counter() - phase_start < prepare_s:
                frame = buffer.take(0.25)
                if capture.error:
                    raise capture.error
                if frame is None:
                    continue
                remaining = prepare_s - (time.perf_counter() - phase_start)
                if _show_frame(
                    frame.image_bgr,
                    "START IN",
                    step.prompt,
                    remaining,
                    step_index,
                ) == 27:
                    aborted = True
                    break
            if aborted:
                break

            phase_start = time.perf_counter()
            while time.perf_counter() - phase_start < step.duration_s:
                frame = buffer.take(0.25)
                if capture.error:
                    raise capture.error
                if frame is None:
                    continue
                signal = backend.analyze(frame.image_bgr, frame.arrived_ns)
                samples[step.name].append(signal)
                remaining = step.duration_s - (time.perf_counter() - phase_start)
                if _show_frame(
                    frame.image_bgr,
                    "MESSUNG",
                    step.prompt,
                    remaining,
                    step_index,
                    signal,
                ) == 27:
                    aborted = True
                    break
            if aborted:
                break
    finally:
        capture.stop()
        backend.close()
        cv2.destroyAllWindows()

    report: dict[str, object] = {
        "privacy": "Aggregate facial signals only; no images or video stored",
        "output": "Disabled",
        "aborted": aborted,
        "camera": {
            "requested": [camera_config.width, camera_config.height, camera_config.fps],
            "actual": [capture.actual_width, capture.actual_height, capture.actual_fps],
            "requested_fourcc": camera_config.fourcc.upper(),
            "actual_fourcc": capture.actual_fourcc,
            "requested_exposure": camera_config.exposure,
            "actual_exposure": capture.actual_exposure,
        },
        "protocol": [asdict(step) for step in PROTOCOL],
        "steps": {name: _summarize(values) for name, values in samples.items()},
    }
    if export_path is not None:
        import json

        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _summarize(samples: list[FaceSignals]) -> dict[str, object]:
    present = [sample for sample in samples if sample.face_present]
    result: dict[str, object] = {
        "frames": len(samples),
        "face_present_rate": len(present) / max(len(samples), 1),
    }
    fields = (
        "person_left_eye_closed",
        "person_right_eye_closed",
        "person_left_eye_aspect_ratio",
        "person_right_eye_aspect_ratio",
        "jaw_open",
        "brow_inner_up",
        "brow_outer_up_left",
        "brow_outer_up_right",
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
    )
    for field in fields:
        values = np.asarray([getattr(sample, field) for sample in present], dtype=float)
        if values.size:
            result[field] = {
                "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    if present:
        asymmetry = np.asarray(
            [sample.person_left_eye_closed - sample.person_right_eye_closed for sample in present]
        )
        result["left_minus_right_eye"] = {
            "median": float(np.median(asymmetry)),
            "p05": float(np.percentile(asymmetry, 5)),
            "p95": float(np.percentile(asymmetry, 95)),
            "min": float(np.min(asymmetry)),
            "max": float(np.max(asymmetry)),
        }
    return result


def _show_frame(
    image: object,
    phase: str,
    prompt: str,
    remaining_s: float,
    step_index: int,
    signal: FaceSignals | None = None,
) -> int:
    import cv2

    display = image.copy()
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    lines = (
        f"{phase}  {step_index + 1}/{len(PROTOCOL)}  {max(remaining_s, 0):.1f}s",
        prompt,
        "Leertaste = Start | Esc = Abbruch | AUSGABE DEAKTIVIERT",
    )
    for index, line in enumerate(lines):
        color = (0, 255, 255) if index == 0 else (255, 255, 255)
        cv2.putText(display, line, (24, 36 + index * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2)
    if signal is not None:
        status = (
            f"L/R {signal.person_left_eye_closed:.2f}/{signal.person_right_eye_closed:.2f}  "
            f"Mund {signal.jaw_open:.2f}  Braue {signal.brow_inner_up:.2f}"
        )
        cv2.putText(display, status, (24, display.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(WINDOW_TITLE, display)
    return cv2.waitKey(1) & 0xFF
