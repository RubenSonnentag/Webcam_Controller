from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .backends.base import FaceBackend
from .capture import CameraCapture
from .config import CameraConfig
from .latest_frame import LatestFrameBuffer
from .mouth import evaluate_mouth_trials
from .sweep import evaluate_negative_stream, run_parameter_sweep
from .trial_test import (
    EAR_CLOSE_THRESHOLDS,
    Trial,
    _classify_trial,
    _summarize_trials,
    _threshold_key,
    _trials,
)
from .types import FaceSignals


RECORD_WINDOW = "Lokale Gestenaufnahme (sensible Gesichtsdaten)"
POSE_GATE_THRESHOLDS_DEG = (10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0)


def record_dataset(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    output_dir: Path,
    trials_per_gesture: int = 10,
    trial_groups: tuple[str, ...] | None = None,
    roi_size: int = 512,
    countdown_s: float = 1.5,
    measurement_s: float = 2.5,
    auto_advance: bool = False,
    group_change_s: float = 4.0,
    trials_override: tuple[Trial, ...] | None = None,
    dataset_kind: str = "controlled_trials",
) -> dict[str, object]:
    import cv2

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Dataset directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "face_roi.avi"
    manifest_path = output_dir / "manifest.json"
    trials = list(trials_override) if trials_override is not None else _trials(
        trials_per_gesture, trial_groups
    )
    if not trials:
        raise ValueError("At least one valid trial group is required")
    if countdown_s < 0 or group_change_s < 0:
        raise ValueError("Preparation durations must not be negative")

    buffer = LatestFrameBuffer()
    capture = CameraCapture(camera_config, buffer)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        camera_config.fps,
        (roi_size, roi_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")

    frames: list[dict[str, object]] = []
    aborted = False
    first_capture_ns: int | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    capture.start()
    try:
        cv2.namedWindow(RECORD_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RECORD_WINDOW, 1280, 720)
        cv2.setWindowProperty(RECORD_WINDOW, cv2.WND_PROP_TOPMOST, 1)
        previous_expected: str | None = None
        for trial_index, trial in enumerate(trials):
            accent = _trial_color(trial)
            group_changed = trial.expected != previous_expected
            if not auto_advance or trial_index == 0:
                if not _await_space(buffer, capture, trial, trial_index, len(trials)):
                    aborted = True
                    break
            if auto_advance:
                phase = "NEUE AKTION IN" if group_changed else "NAECHSTER VERSUCH IN"
                preparation_s = group_change_s if group_changed else countdown_s
            else:
                phase = "START IN"
                preparation_s = countdown_s
            if not _countdown(
                buffer,
                capture,
                trial,
                trial_index,
                len(trials),
                preparation_s,
                phase=phase,
                accent=accent,
            ):
                aborted = True
                break
            started = time.perf_counter()
            while time.perf_counter() - started < measurement_s:
                packet = buffer.take(0.25)
                if capture.error:
                    raise capture.error
                if packet is None:
                    continue
                signals = backend.analyze(packet.image_bgr, packet.arrived_ns)
                if signals.face_bbox_normalized is not None:
                    last_bbox = signals.face_bbox_normalized
                roi = _crop_face_roi(packet.image_bgr, last_bbox, roi_size)
                writer.write(roi)
                if first_capture_ns is None:
                    first_capture_ns = packet.arrived_ns
                frames.append(
                    {
                        "frame": len(frames),
                        "relative_ms": (packet.arrived_ns - first_capture_ns) / 1e6,
                        "expected": trial.expected,
                        "trial_number": trial.number,
                        "face_present": signals.face_present,
                    }
                )
                remaining = measurement_s - (time.perf_counter() - started)
                if _show_recording(
                    packet.image_bgr,
                    trial,
                    trial_index,
                    len(trials),
                    remaining,
                    accent=accent,
                ) == 27:
                    aborted = True
                    break
            if aborted:
                break
            previous_expected = trial.expected
    finally:
        capture.stop()
        backend.close()
        writer.release()
        cv2.destroyAllWindows()

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_kind": dataset_kind,
        "privacy_notice": "Sensitive facial ROI video; local storage only",
        "aborted": aborted,
        "video": video_path.name,
        "video_sha256": _sha256(video_path),
        "encoding": "MJPG AVI, face ROI only",
        "roi_size": [roi_size, roi_size],
        "requested_camera": [camera_config.width, camera_config.height, camera_config.fps],
        "actual_camera": [capture.actual_width, capture.actual_height, capture.actual_fps],
        "requested_fourcc": camera_config.fourcc.upper(),
        "actual_fourcc": capture.actual_fourcc,
        "requested_exposure": camera_config.exposure,
        "actual_exposure": capture.actual_exposure,
        "frame_count": len(frames),
        "auto_advance": auto_advance,
        "group_change_seconds": group_change_s if auto_advance else None,
        "frames": frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset": str(output_dir),
        "aborted": aborted,
        "frame_count": len(frames),
        "video_sha256": manifest["video_sha256"],
        "privacy": manifest["privacy_notice"],
    }


def record_mouth_dataset(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    output_dir: Path,
    trials_per_group: int = 10,
    roi_size: int = 512,
    group_change_s: float = 4.0,
) -> dict[str, object]:
    if trials_per_group < 1:
        raise ValueError("trials_per_group must be positive")
    trials = tuple(
        Trial(
            "mouth_neutral",
            "Mund entspannt geschlossen halten; nicht sprechen",
            number,
            trials_per_group,
        )
        for number in range(1, trials_per_group + 1)
    ) + tuple(
        Trial(
            "mouth_open",
            "Mund EINMAL deutlich oeffnen, kurz halten und wieder schliessen",
            number,
            trials_per_group,
        )
        for number in range(1, trials_per_group + 1)
    )
    return record_dataset(
        backend,
        camera_config,
        output_dir=output_dir,
        roi_size=roi_size,
        auto_advance=True,
        group_change_s=group_change_s,
        trials_override=trials,
        dataset_kind="mouth_trials",
    )


def record_negative_dataset(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    output_dir: Path,
    duration_s: float = 120.0,
    roi_size: int = 512,
    countdown_s: float = 3.0,
) -> dict[str, object]:
    import cv2

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Dataset directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "face_roi.avi"
    manifest_path = output_dir / "manifest.json"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        camera_config.fps,
        (roi_size, roi_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")

    trial = Trial(
        "negative_stream",
        "Natuerlich schauen und blinzeln; keine absichtliche einseitige Geste",
        1,
        1,
    )
    buffer = LatestFrameBuffer()
    capture = CameraCapture(camera_config, buffer)
    frames: list[dict[str, object]] = []
    first_capture_ns: int | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    aborted = False
    capture.start()
    try:
        cv2.namedWindow(RECORD_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RECORD_WINDOW, 1280, 720)
        cv2.setWindowProperty(RECORD_WINDOW, cv2.WND_PROP_TOPMOST, 1)
        if not _await_space(buffer, capture, trial, 0, 1):
            aborted = True
        elif not _countdown(buffer, capture, trial, 0, 1, countdown_s):
            aborted = True
        else:
            started = time.perf_counter()
            while time.perf_counter() - started < duration_s:
                packet = buffer.take(0.25)
                if capture.error:
                    raise capture.error
                if packet is None:
                    continue
                signals = backend.analyze(packet.image_bgr, packet.arrived_ns)
                if signals.face_bbox_normalized is not None:
                    last_bbox = signals.face_bbox_normalized
                writer.write(_crop_face_roi(packet.image_bgr, last_bbox, roi_size))
                if first_capture_ns is None:
                    first_capture_ns = packet.arrived_ns
                frames.append(
                    {
                        "frame": len(frames),
                        "relative_ms": (packet.arrived_ns - first_capture_ns) / 1e6,
                        "expected": trial.expected,
                        "trial_number": 1,
                        "face_present": signals.face_present,
                    }
                )
                remaining = duration_s - (time.perf_counter() - started)
                if _show_recording(packet.image_bgr, trial, 0, 1, remaining) == 27:
                    aborted = True
                    break
    finally:
        capture.stop()
        backend.close()
        writer.release()
        cv2.destroyAllWindows()

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_kind": "negative_stream",
        "privacy_notice": "Sensitive facial ROI video; local storage only",
        "aborted": aborted,
        "video": video_path.name,
        "video_sha256": _sha256(video_path),
        "encoding": "MJPG AVI, face ROI only",
        "roi_size": [roi_size, roi_size],
        "requested_camera": [camera_config.width, camera_config.height, camera_config.fps],
        "actual_camera": [capture.actual_width, capture.actual_height, capture.actual_fps],
        "requested_fourcc": camera_config.fourcc.upper(),
        "actual_fourcc": capture.actual_fourcc,
        "requested_exposure": camera_config.exposure,
        "actual_exposure": capture.actual_exposure,
        "requested_duration_s": duration_s,
        "frame_count": len(frames),
        "frames": frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset": str(output_dir),
        "aborted": aborted,
        "frame_count": len(frames),
        "video_sha256": manifest["video_sha256"],
        "privacy": manifest["privacy_notice"],
    }


def record_pose_dataset(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    output_dir: Path,
    seconds_per_pose: float = 12.0,
    roi_size: int = 512,
    countdown_s: float = 2.0,
    poses: tuple[Trial, ...] | None = None,
    dataset_kind: str = "pose_negative",
) -> dict[str, object]:
    """Record natural blinks at neutral and deliberately rotated head poses."""
    import cv2

    if seconds_per_pose <= 0:
        raise ValueError("seconds_per_pose must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Dataset directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "face_roi.avi"
    manifest_path = output_dir / "manifest.json"
    poses = poses or (
        Trial("pose_neutral", "Geradeaus schauen und 3-5 Mal normal blinzeln", 1, 1),
        Trial("pose_left", "Kopf nach LINKS drehen, halten, 3-5 Mal normal blinzeln", 1, 1),
        Trial("pose_right", "Kopf nach RECHTS drehen, halten, 3-5 Mal normal blinzeln", 1, 1),
        Trial("pose_up", "Kopf nach OBEN neigen, halten, 3-5 Mal normal blinzeln", 1, 1),
        Trial("pose_down", "Kopf nach UNTEN neigen, halten, 3-5 Mal normal blinzeln", 1, 1),
    )
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        camera_config.fps,
        (roi_size, roi_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")

    buffer = LatestFrameBuffer()
    capture = CameraCapture(camera_config, buffer)
    frames: list[dict[str, object]] = []
    first_capture_ns: int | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    aborted = False
    capture.start()
    try:
        cv2.namedWindow(RECORD_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RECORD_WINDOW, 1280, 720)
        cv2.setWindowProperty(RECORD_WINDOW, cv2.WND_PROP_TOPMOST, 1)
        for pose_index, pose in enumerate(poses):
            if not _await_space(buffer, capture, pose, pose_index, len(poses)):
                aborted = True
                break
            if not _countdown(
                buffer, capture, pose, pose_index, len(poses), countdown_s
            ):
                aborted = True
                break
            started = time.perf_counter()
            while time.perf_counter() - started < seconds_per_pose:
                packet = buffer.take(0.25)
                if capture.error:
                    raise capture.error
                if packet is None:
                    continue
                signals = backend.analyze(packet.image_bgr, packet.arrived_ns)
                if signals.face_bbox_normalized is not None:
                    last_bbox = signals.face_bbox_normalized
                writer.write(_crop_face_roi(packet.image_bgr, last_bbox, roi_size))
                if first_capture_ns is None:
                    first_capture_ns = packet.arrived_ns
                frames.append(
                    {
                        "frame": len(frames),
                        "relative_ms": (packet.arrived_ns - first_capture_ns) / 1e6,
                        "expected": pose.expected,
                        "trial_number": 1,
                        "face_present": signals.face_present,
                    }
                )
                remaining = seconds_per_pose - (time.perf_counter() - started)
                if _show_recording(
                    packet.image_bgr, pose, pose_index, len(poses), remaining
                ) == 27:
                    aborted = True
                    break
            if aborted:
                break
    finally:
        capture.stop()
        backend.close()
        writer.release()
        cv2.destroyAllWindows()

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_kind": dataset_kind,
        "privacy_notice": "Sensitive facial ROI video; local storage only",
        "aborted": aborted,
        "video": video_path.name,
        "video_sha256": _sha256(video_path),
        "encoding": "MJPG AVI, face ROI only",
        "roi_size": [roi_size, roi_size],
        "requested_camera": [camera_config.width, camera_config.height, camera_config.fps],
        "actual_camera": [capture.actual_width, capture.actual_height, capture.actual_fps],
        "requested_fourcc": camera_config.fourcc.upper(),
        "actual_fourcc": capture.actual_fourcc,
        "requested_exposure": camera_config.exposure,
        "actual_exposure": capture.actual_exposure,
        "seconds_per_pose": seconds_per_pose,
        "pose_order": [pose.expected for pose in poses],
        "frame_count": len(frames),
        "frames": frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset": str(output_dir),
        "aborted": aborted,
        "frame_count": len(frames),
        "video_sha256": manifest["video_sha256"],
        "privacy": manifest["privacy_notice"],
    }


def record_pose_sweep_dataset(
    backend: FaceBackend,
    camera_config: CameraConfig,
    *,
    output_dir: Path,
    seconds_per_step: float = 18.0,
    roi_size: int = 512,
) -> dict[str, object]:
    poses = (
        Trial(
            "pose_neutral",
            "Geradeaus und ruhig schauen; 3-5 Mal normal blinzeln",
            1,
            1,
        ),
        Trial(
            "pose_yaw_sweep",
            "Langsam: Mitte > LINKS > Mitte > RECHTS > Mitte; zweimal",
            1,
            1,
        ),
        Trial(
            "pose_pitch_sweep",
            "Langsam: Mitte > OBEN > Mitte > UNTEN > Mitte; zweimal",
            1,
            1,
        ),
    )
    return record_pose_dataset(
        backend,
        camera_config,
        output_dir=output_dir,
        seconds_per_pose=seconds_per_step,
        roi_size=roi_size,
        poses=poses,
        dataset_kind="pose_sweep",
    )


def recover_pose_sweep_manifest(
    dataset_dir: Path,
    *,
    requested_camera: tuple[int, int, float] = (1920, 1080, 30.0),
) -> dict[str, object]:
    """Recover labels for a completed constant-frame-rate sweep video."""
    import cv2

    video_path = dataset_dir / "face_roi.avi"
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open dataset video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        roi_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        roi_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count < 3 or fps <= 0:
        raise ValueError("Video metadata is insufficient for recovery")

    labels = ("pose_neutral", "pose_yaw_sweep", "pose_pitch_sweep")
    frames: list[dict[str, object]] = []
    for index in range(frame_count):
        section = min(index * len(labels) // frame_count, len(labels) - 1)
        frames.append(
            {
                "frame": index,
                "relative_ms": index * 1000.0 / fps,
                "expected": labels[section],
                "trial_number": 1,
                "face_present": None,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_kind": "pose_sweep",
        "privacy_notice": "Sensitive facial ROI video; local storage only",
        "aborted": False,
        "recovered_manifest": True,
        "timing_source": "constant frame rate reconstructed from AVI metadata",
        "video": video_path.name,
        "video_sha256": _sha256(video_path),
        "encoding": "MJPG AVI, face ROI only",
        "roi_size": [roi_width, roi_height],
        "requested_camera": list(requested_camera),
        "actual_camera": [requested_camera[0], requested_camera[1], fps],
        "seconds_per_pose": frame_count / fps / len(labels),
        "pose_order": list(labels),
        "frame_count": frame_count,
        "frames": frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset": str(dataset_dir),
        "frame_count": frame_count,
        "fps": fps,
        "video_sha256": manifest["video_sha256"],
        "recovered_manifest": True,
    }


def replay_dataset(
    backend: FaceBackend,
    *,
    dataset_dir: Path,
    export_path: Path | None,
    run_sweep: bool = False,
) -> dict[str, object]:
    import cv2

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema")
    video_path = dataset_dir / str(manifest["video"])
    expected_hash = str(manifest["video_sha256"])
    actual_hash = _sha256(video_path)
    if actual_hash != expected_hash:
        raise ValueError("Dataset video hash does not match manifest")
    labels = list(manifest["frames"])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open dataset video: {video_path}")
    grouped: dict[tuple[str, int], list[FaceSignals]] = {}
    inference_ms: list[float] = []
    read_frames = 0
    try:
        for label in labels:
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {read_frames}, expected {len(labels)}")
            started_ns = time.perf_counter_ns()
            signals = backend.analyze(image, int(float(label["relative_ms"]) * 1e6))
            inference_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
            key = (str(label["expected"]), int(label["trial_number"]))
            grouped.setdefault(key, []).append(signals)
            read_frames += 1
        if capture.read()[0]:
            raise RuntimeError("Video contains more frames than the manifest")
    finally:
        capture.release()
        backend.close()

    dataset_kind = str(manifest.get("dataset_kind", "controlled_trials"))
    report: dict[str, object] = {
        "privacy": "Local replay; no images exported",
        "output": "Disabled",
        "dataset_kind": dataset_kind,
        "dataset_video_sha256": actual_hash,
        "frame_count": read_frames,
        "inference_ms": _stats(inference_ms),
    }
    if dataset_kind == "negative_stream":
        samples = next(iter(grouped.values()), [])
        report["negative_evaluation"] = evaluate_negative_stream(samples)
    elif dataset_kind == "mouth_trials":
        report["mouth_evaluation"] = evaluate_mouth_trials(grouped)
    elif dataset_kind in ("pose_negative", "pose_sweep"):
        neutral_samples = grouped.get(("pose_neutral", 1), [])
        visible_neutral = [sample for sample in neutral_samples if sample.face_present]
        if not visible_neutral:
            raise ValueError("Pose dataset has no visible neutral-face samples")
        neutral_yaw = float(np.median([sample.yaw_deg for sample in visible_neutral]))
        neutral_pitch = float(np.median([sample.pitch_deg for sample in visible_neutral]))
        pose_steps: dict[str, object] = {}
        for (label, _), samples in grouped.items():
            step: dict[str, object] = {
                "pose": _pose_summary(samples),
                "without_pose_gate": evaluate_negative_stream(samples),
                "with_pose_gate": evaluate_negative_stream(
                    samples,
                    neutral_yaw_deg=neutral_yaw,
                    neutral_pitch_deg=neutral_pitch,
                ),
            }
            if dataset_kind == "pose_sweep":
                step["gate_sensitivity"] = {
                    str(int(limit)): {
                        key: value
                        for key, value in evaluate_negative_stream(
                            samples,
                            neutral_yaw_deg=neutral_yaw,
                            neutral_pitch_deg=neutral_pitch,
                            max_yaw_delta_deg=limit,
                            max_pitch_delta_deg=limit,
                            yaw_reenable_delta_deg=max(limit - 3.0, 0.0),
                            pitch_reenable_delta_deg=max(limit - 3.0, 0.0),
                        ).items()
                        if key
                        in (
                            "left_false_events",
                            "right_false_events",
                            "total_false_events",
                            "pose_blocked_frame_rate",
                        )
                    }
                    for limit in POSE_GATE_THRESHOLDS_DEG
                }
            pose_steps[label] = step
        pose_evaluation: dict[str, object] = {
            "neutral_reference_deg": {"yaw": neutral_yaw, "pitch": neutral_pitch},
            "gate_limits_deg": {
                "yaw_block": 15.0,
                "yaw_reenable": 12.0,
                "pitch_block": 18.0,
                "pitch_reenable": 15.0,
                "reenable_stable_ms": 150.0,
            },
            "steps": pose_steps,
        }
        if dataset_kind == "pose_sweep":
            pose_evaluation["threshold_sweep"] = {
                label: _pose_threshold_sweep(samples, neutral_yaw, neutral_pitch)
                for (label, _), samples in grouped.items()
            }
        report["pose_evaluation"] = pose_evaluation
    else:
        results = [
            _classify_trial(Trial(expected, "", number), samples)
            for (expected, number), samples in grouped.items()
        ]
        report.update(
            {
                "blendshape_summary": _summarize_trials(results),
                "ear_threshold_summaries": {
                    _threshold_key(threshold): _summarize_trials(
                        results,
                        decision_field=f"ear_decisions_{_threshold_key(threshold)}",
                    )
                    for threshold in EAR_CLOSE_THRESHOLDS
                },
                "trials": results,
            }
        )
        if run_sweep:
            report["parameter_sweep"] = run_parameter_sweep(grouped)
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _crop_face_roi(
    image: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    size: int,
) -> np.ndarray:
    import cv2

    if bbox is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    left, top, right, bottom = bbox
    center_x = (left + right) * 0.5 * width
    center_y = (top + bottom) * 0.5 * height
    side = max((right - left) * width, (bottom - top) * height) * 1.45
    side = max(side, 32.0)
    x0 = max(int(center_x - side / 2), 0)
    y0 = max(int(center_y - side / 2), 0)
    x1 = min(int(center_x + side / 2), width)
    y1 = min(int(center_y + side / 2), height)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _await_space(
    buffer: LatestFrameBuffer,
    capture: CameraCapture,
    trial: Trial,
    index: int,
    total: int,
) -> bool:
    while True:
        packet = buffer.take(0.25)
        if capture.error:
            raise capture.error
        if packet is None:
            continue
        key = _show(packet.image_bgr, "BEREIT? LEERTASTE", trial, index, total, 0.0)
        if key == 27:
            return False
        if key == 32:
            return True


def _countdown(
    buffer: LatestFrameBuffer,
    capture: CameraCapture,
    trial: Trial,
    index: int,
    total: int,
    seconds: float,
    *,
    phase: str = "START IN",
    accent: tuple[int, int, int] = (0, 255, 255),
) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        packet = buffer.take(0.25)
        if capture.error:
            raise capture.error
        if packet is None:
            continue
        remaining = seconds - (time.perf_counter() - started)
        if _show(
            packet.image_bgr,
            phase,
            trial,
            index,
            total,
            remaining,
            accent=accent,
        ) == 27:
            return False
    return True


def _show(
    image: object,
    phase: str,
    trial: Trial,
    index: int,
    total: int,
    remaining: float,
    *,
    accent: tuple[int, int, int] = (0, 255, 255),
) -> int:
    import cv2

    display = image.copy()
    cv2.rectangle(display, (0, 0), (display.shape[1], 160), (0, 0, 0), -1)
    lines = (
        f"{phase} | Versuch {index + 1}/{total} | {max(remaining, 0):.1f}s",
        trial.prompt,
        "Leertaste = Start | Esc = Abbruch | sensible lokale Aufnahme",
    )
    for row, line in enumerate(lines):
        color = accent if row < 2 else (255, 255, 255)
        cv2.putText(
            display,
            line,
            (24, 40 + row * 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            color,
            2,
        )
    cv2.imshow(RECORD_WINDOW, display)
    return cv2.waitKey(1) & 0xFF


def _show_recording(
    image: object,
    trial: Trial,
    index: int,
    total: int,
    remaining: float,
    *,
    accent: tuple[int, int, int] = (255, 255, 255),
) -> int:
    import cv2

    display = image.copy()
    cv2.rectangle(display, (0, 0), (display.shape[1], 170), (0, 0, 0), -1)
    cv2.circle(display, (38, 38), 15, (0, 0, 255), -1)
    lines = (
        f"AUFNAHME | Versuch {index + 1}/{total} | {max(remaining, 0):.1f}s",
        trial.prompt,
        "Gespeichert wird nur die Gesichts-ROI | Esc = Abbruch",
    )
    for row, line in enumerate(lines):
        color = (0, 0, 255) if row == 0 else accent if row == 1 else (255, 255, 255)
        cv2.putText(
            display,
            line,
            (68 if row == 0 else 24, 46 + row * 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            color,
            2,
        )
    cv2.imshow(RECORD_WINDOW, display)
    return cv2.waitKey(1) & 0xFF


def _trial_color(trial: Trial) -> tuple[int, int, int]:
    return {
        "normal_blink": (0, 255, 255),
        "left_eye": (0, 255, 0),
        "right_eye": (255, 0, 255),
        "mouth_neutral": (255, 255, 255),
        "mouth_open": (255, 255, 0),
    }.get(trial.expected, (255, 255, 255))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _pose_summary(samples: list[FaceSignals]) -> dict[str, object]:
    visible = [sample for sample in samples if sample.face_present]
    return {
        "frame_count": len(samples),
        "face_present_rate": len(visible) / len(samples) if samples else 0.0,
        "yaw_deg": _stats([sample.yaw_deg for sample in visible]),
        "pitch_deg": _stats([sample.pitch_deg for sample in visible]),
        "roll_deg": _stats([sample.roll_deg for sample in visible]),
    }


def _pose_threshold_sweep(
    samples: list[FaceSignals], neutral_yaw: float, neutral_pitch: float
) -> dict[str, dict[str, float]]:
    visible = [sample for sample in samples if sample.face_present]
    if not visible:
        return {"yaw_blocked_rate": {}, "pitch_blocked_rate": {}}
    return {
        "yaw_blocked_rate": {
            str(int(limit)): sum(
                abs(sample.yaw_deg - neutral_yaw) > limit for sample in visible
            )
            / len(visible)
            for limit in POSE_GATE_THRESHOLDS_DEG
        },
        "pitch_blocked_rate": {
            str(int(limit)): sum(
                abs(sample.pitch_deg - neutral_pitch) > limit for sample in visible
            )
            / len(visible)
            for limit in POSE_GATE_THRESHOLDS_DEG
        },
    }
