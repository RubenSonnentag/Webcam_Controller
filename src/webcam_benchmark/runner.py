from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .backends.base import FaceBackend
from .capture import CameraCapture
from .config import CameraConfig
from .gesture import GestureEngine
from .latest_frame import LatestFrameBuffer
from .metrics import RollingMetrics


def run_camera(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    duration_s: float | None,
    export_path: Path | None,
) -> dict[str, object]:
    import cv2
    import psutil

    buffer = LatestFrameBuffer()
    capture = CameraCapture(camera_config, buffer)
    gestures = GestureEngine()
    metrics = RollingMetrics()
    process = psutil.Process()
    logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
    process.cpu_percent(None)
    psutil.cpu_percent(None)
    start_ns = time.perf_counter_ns()
    active_start_ns: int | None = None
    next_analysis_frame_ns = 0
    next_resource_sample_ns = start_ns + 1_000_000_000
    analysis_period_ns = int(1_000_000_000 / camera_config.analysis_rate)
    rate_tolerance_ns = 1_000_000
    rate_limit_enabled = camera_config.analysis_rate < camera_config.fps - 0.1
    analyzed = 0
    faces_present = 0
    latest_signals = None

    capture.start()
    try:
        while True:
            now_ns = time.perf_counter_ns()
            frame = buffer.take(0.25)
            if capture.error:
                raise capture.error
            if frame is None:
                if buffer.closed:
                    break
                continue
            if active_start_ns is None:
                active_start_ns = frame.arrived_ns
            if duration_s is not None and (frame.arrived_ns - active_start_ns) / 1e9 >= duration_s:
                break
            now_ns = time.perf_counter_ns()
            if now_ns >= next_resource_sample_ns:
                metrics.add("process_cpu_total_percent", process.cpu_percent(None) / logical_cpus)
                metrics.add("system_cpu_percent", psutil.cpu_percent(None))
                metrics.add("memory_mb", process.memory_info().rss / (1024 * 1024))
                next_resource_sample_ns = now_ns + 1_000_000_000
            if rate_limit_enabled and frame.arrived_ns + rate_tolerance_ns < next_analysis_frame_ns:
                continue

            inference_start_ns = time.perf_counter_ns()
            metrics.add("frame_age_at_inference_ms", (inference_start_ns - frame.arrived_ns) / 1e6)
            latest_signals = backend.analyze(frame.image_bgr, frame.arrived_ns)
            if latest_signals.face_present:
                faces_present += 1
                metrics.add("yaw_deg", latest_signals.yaw_deg)
                metrics.add("pitch_deg", latest_signals.pitch_deg)
                metrics.add("roll_deg", latest_signals.roll_deg)
            inference_end_ns = time.perf_counter_ns()
            events = gestures.update(latest_signals)
            gesture_end_ns = time.perf_counter_ns()

            # Output-free mapping placeholder: event names only, no OS interaction.
            hypothetical_actions = [event.name for event in events if event.active]
            mapping_end_ns = time.perf_counter_ns()
            _ = hypothetical_actions

            metrics.add("inference_ms", (inference_end_ns - inference_start_ns) / 1e6)
            metrics.add("gesture_ms", (gesture_end_ns - inference_end_ns) / 1e6)
            metrics.add("mapping_ms", (mapping_end_ns - gesture_end_ns) / 1e6)
            metrics.add("capture_to_inference_ms", (inference_end_ns - frame.arrived_ns) / 1e6)
            metrics.add("software_ms", (mapping_end_ns - frame.arrived_ns) / 1e6)
            analyzed += 1
            next_analysis_frame_ns = frame.arrived_ns + analysis_period_ns

            if camera_config.preview:
                display = frame.image_bgr.copy()
                _draw_status(display, latest_signals, metrics.snapshot(), analyzed)
                cv2.imshow("Phase-2 Benchmark (no output)", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        capture.stop()
        backend.close()
        cv2.destroyAllWindows()

    wall_elapsed_s = max((time.perf_counter_ns() - start_ns) / 1e9, 1e-9)
    memory_mb = process.memory_info().rss / (1024 * 1024)
    frame_stats = buffer.stats()
    capture_span_s = max(
        (frame_stats.last_arrived_ns - frame_stats.first_arrived_ns) / 1e9,
        1e-9,
    )
    measured_capture_rate = max(frame_stats.published - 1, 0) / capture_span_s
    measured_analysis_rate = analyzed / capture_span_s
    report: dict[str, object] = {
        "privacy": "No images or video stored",
        "output": "Disabled",
        "camera": {
            "requested": [camera_config.width, camera_config.height, camera_config.fps],
            "actual": [capture.actual_width, capture.actual_height, capture.actual_fps],
            "measured_rate": measured_capture_rate,
        },
        "analysis": {
            "requested_rate": camera_config.analysis_rate,
            "actual_rate": measured_analysis_rate,
            "count": analyzed,
            "face_present_count": faces_present,
            "face_present_rate": faces_present / max(analyzed, 1),
        },
        "frame_buffer": asdict(frame_stats),
        "memory_mb": memory_mb,
        "wall_elapsed_s": wall_elapsed_s,
        "metrics_ms": metrics.snapshot(),
    }
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _draw_status(image: object, signals: object, snapshot: dict[str, object], analyzed: int) -> None:
    import cv2

    lines = ["OUTPUT DISABLED", f"Analyzed: {analyzed}"]
    if signals is not None:
        lines.extend(
            [
                f"L/R eye: {signals.person_left_eye_closed:.2f} / {signals.person_right_eye_closed:.2f}",
                f"Yaw/Pitch/Roll: {signals.yaw_deg:.1f} / {signals.pitch_deg:.1f} / {signals.roll_deg:.1f}",
            ]
        )
    software = snapshot.get("software_ms", {})
    if software:
        lines.append(f"Software median/p95: {software.get('median', 0):.1f}/{software.get('p95', 0):.1f} ms")
    for index, line in enumerate(lines):
        cv2.putText(image, line, (20, 35 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
