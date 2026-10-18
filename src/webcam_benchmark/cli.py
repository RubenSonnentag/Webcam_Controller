from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import MediaPipeFaceBackend
from .config import CameraConfig
from .dataset import (
    record_dataset,
    record_negative_dataset,
    record_pose_dataset,
    record_pose_sweep_dataset,
    recover_pose_sweep_manifest,
    replay_dataset,
)
from .eye_test import run_eye_test
from .guided import run_guided
from .runner import run_camera
from .synthetic import run_synthetic
from .trial_test import run_trial_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Output-free facial gesture benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Run without camera/model")
    synthetic.add_argument("--seconds", type=float, default=5.0)
    synthetic.add_argument("--rate", type=float, default=60.0)

    camera = subparsers.add_parser("camera", help="Run MediaPipe camera benchmark")
    camera.add_argument("--model", type=Path, required=True)
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--width", type=int, default=1920)
    camera.add_argument("--height", type=int, default=1080)
    camera.add_argument("--fps", type=float, default=60.0)
    camera.add_argument("--capture-backend", choices=("auto", "msmf", "dshow"), default="auto")
    camera.add_argument("--fourcc", choices=("auto", "mjpg", "yuy2"), default="auto")
    camera.add_argument("--exposure", type=float)
    camera.add_argument("--analysis-rate", type=float, choices=(30.0, 45.0, 60.0), default=60.0)
    camera.add_argument("--seconds", type=float)
    camera.add_argument("--no-preview", action="store_true")
    camera.add_argument("--export", type=Path)

    guided = subparsers.add_parser("guided", help="Run prompted gesture measurements")
    guided.add_argument("--model", type=Path, required=True)
    guided.add_argument("--camera", type=int, default=0)
    guided.add_argument("--width", type=int, default=1920)
    guided.add_argument("--height", type=int, default=1080)
    guided.add_argument("--fps", type=float, default=30.0)
    guided.add_argument("--capture-backend", choices=("auto", "msmf", "dshow"), default="dshow")
    guided.add_argument("--fourcc", choices=("auto", "mjpg", "yuy2"), default="auto")
    guided.add_argument("--exposure", type=float)
    guided.add_argument("--analysis-rate", type=float, choices=(30.0, 45.0, 60.0), default=30.0)
    guided.add_argument("--prepare-seconds", type=float, default=3.0)
    guided.add_argument("--export", type=Path)

    eyes = subparsers.add_parser("eyes", help="Run focused eye timing test")
    eyes.add_argument("--model", type=Path, required=True)
    eyes.add_argument("--camera", type=int, default=0)
    eyes.add_argument("--width", type=int, default=1920)
    eyes.add_argument("--height", type=int, default=1080)
    eyes.add_argument("--fps", type=float, default=30.0)
    eyes.add_argument("--capture-backend", choices=("auto", "msmf", "dshow"), default="dshow")
    eyes.add_argument("--fourcc", choices=("auto", "mjpg", "yuy2"), default="auto")
    eyes.add_argument("--exposure", type=float)
    eyes.add_argument("--analysis-rate", type=float, choices=(30.0, 45.0, 60.0), default=30.0)
    eyes.add_argument("--export", type=Path)

    trials = subparsers.add_parser("trials", help="Run controlled single-gesture eye trials")
    trials.add_argument("--model", type=Path, required=True)
    trials.add_argument("--camera", type=int, default=0)
    trials.add_argument("--width", type=int, default=1920)
    trials.add_argument("--height", type=int, default=1080)
    trials.add_argument("--fps", type=float, default=30.0)
    trials.add_argument("--capture-backend", choices=("auto", "msmf", "dshow"), default="dshow")
    trials.add_argument("--fourcc", choices=("auto", "mjpg", "yuy2"), default="auto")
    trials.add_argument("--exposure", type=float)
    trials.add_argument("--analysis-rate", type=float, choices=(30.0, 45.0, 60.0), default=30.0)
    trials.add_argument("--trials-per-gesture", type=int, default=10)
    trials.add_argument(
        "--trial-groups",
        nargs="+",
        choices=("normal_blink", "left_eye", "right_eye"),
        default=None,
    )
    trials.add_argument("--export", type=Path)

    record = subparsers.add_parser("record", help="Record a local labeled face-ROI dataset")
    record.add_argument("--model", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--camera", type=int, default=0)
    record.add_argument("--width", type=int, default=1920)
    record.add_argument("--height", type=int, default=1080)
    record.add_argument("--fps", type=float, default=30.0)
    record.add_argument("--capture-backend", choices=("auto", "msmf", "dshow"), default="dshow")
    record.add_argument("--fourcc", choices=("auto", "mjpg", "yuy2"), default="auto")
    record.add_argument("--exposure", type=float)
    record.add_argument("--trials-per-gesture", type=int, default=10)
    record.add_argument(
        "--trial-groups",
        nargs="+",
        choices=("normal_blink", "left_eye", "right_eye"),
        default=None,
    )
    record.add_argument("--roi-size", type=int, default=512)

    negative = subparsers.add_parser(
        "record-negative", help="Record continuous neutral behavior and natural blinks"
    )
    negative.add_argument("--model", type=Path, required=True)
    negative.add_argument("--output", type=Path, required=True)
    negative.add_argument("--seconds", type=float, default=120.0)
    negative.add_argument("--camera", type=int, default=0)
    negative.add_argument("--width", type=int, default=1920)
    negative.add_argument("--height", type=int, default=1080)
    negative.add_argument("--fps", type=float, default=30.0)
    negative.add_argument("--capture-backend", choices=("auto", "msmf", "dshow"), default="dshow")
    negative.add_argument("--fourcc", choices=("auto", "mjpg", "yuy2"), default="auto")
    negative.add_argument("--exposure", type=float)
    negative.add_argument("--roi-size", type=int, default=512)

    pose = subparsers.add_parser(
        "record-pose", help="Record natural blinks at neutral and rotated head poses"
    )
    pose.add_argument("--model", type=Path, required=True)
    pose.add_argument("--output", type=Path, required=True)
    pose.add_argument("--seconds-per-pose", type=float, default=12.0)
    pose.add_argument("--camera", type=int, default=0)
    pose.add_argument("--width", type=int, default=1920)
    pose.add_argument("--height", type=int, default=1080)
    pose.add_argument("--fps", type=float, default=30.0)
    pose.add_argument("--capture-backend", choices=("auto", "msmf", "dshow"), default="dshow")
    pose.add_argument("--fourcc", choices=("auto", "mjpg", "yuy2"), default="auto")
    pose.add_argument("--exposure", type=float)
    pose.add_argument("--roi-size", type=int, default=512)

    pose_sweep = subparsers.add_parser(
        "record-pose-sweep", help="Record slow yaw and pitch calibration sweeps"
    )
    pose_sweep.add_argument("--model", type=Path, required=True)
    pose_sweep.add_argument("--output", type=Path, required=True)
    pose_sweep.add_argument("--seconds-per-step", type=float, default=18.0)
    pose_sweep.add_argument("--camera", type=int, default=0)
    pose_sweep.add_argument("--width", type=int, default=1920)
    pose_sweep.add_argument("--height", type=int, default=1080)
    pose_sweep.add_argument("--fps", type=float, default=30.0)
    pose_sweep.add_argument("--capture-backend", choices=("auto", "msmf", "dshow"), default="dshow")
    pose_sweep.add_argument("--fourcc", choices=("auto", "mjpg", "yuy2"), default="auto")
    pose_sweep.add_argument("--exposure", type=float)
    pose_sweep.add_argument("--roi-size", type=int, default=512)

    recover_sweep = subparsers.add_parser(
        "recover-pose-sweep", help="Recover a missing manifest from a completed sweep AVI"
    )
    recover_sweep.add_argument("--dataset", type=Path, required=True)
    recover_sweep.add_argument("--camera-width", type=int, default=1920)
    recover_sweep.add_argument("--camera-height", type=int, default=1080)
    recover_sweep.add_argument("--camera-fps", type=float, default=30.0)

    replay = subparsers.add_parser("replay", help="Replay a recorded local dataset")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--dataset", type=Path, required=True)
    replay.add_argument("--export", type=Path)
    replay.add_argument("--sweep", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "synthetic":
        report = run_synthetic(args.seconds, args.rate)
    elif args.command == "camera":
        report = run_camera(
            MediaPipeFaceBackend(args.model),
            CameraConfig(
                index=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                analysis_rate=args.analysis_rate,
                preview=not args.no_preview,
                backend=args.capture_backend,
                fourcc=args.fourcc,
                exposure=args.exposure,
            ),
            duration_s=args.seconds,
            export_path=args.export,
        )
    elif args.command == "guided":
        report = run_guided(
            MediaPipeFaceBackend(args.model),
            CameraConfig(
                index=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                analysis_rate=args.analysis_rate,
                preview=True,
                backend=args.capture_backend,
                fourcc=args.fourcc,
                exposure=args.exposure,
            ),
            prepare_s=args.prepare_seconds,
            export_path=args.export,
        )
    elif args.command == "eyes":
        report = run_eye_test(
            MediaPipeFaceBackend(args.model),
            CameraConfig(
                index=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                analysis_rate=args.analysis_rate,
                preview=True,
                backend=args.capture_backend,
                fourcc=args.fourcc,
                exposure=args.exposure,
            ),
            export_path=args.export,
        )
    elif args.command == "trials":
        report = run_trial_test(
            MediaPipeFaceBackend(args.model),
            CameraConfig(
                index=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                analysis_rate=args.analysis_rate,
                preview=True,
                backend=args.capture_backend,
                fourcc=args.fourcc,
                exposure=args.exposure,
            ),
            export_path=args.export,
            trials_per_gesture=args.trials_per_gesture,
            trial_groups=tuple(args.trial_groups) if args.trial_groups else None,
        )
    elif args.command == "record":
        report = record_dataset(
            MediaPipeFaceBackend(args.model),
            CameraConfig(
                index=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                analysis_rate=args.fps,
                preview=True,
                backend=args.capture_backend,
                fourcc=args.fourcc,
                exposure=args.exposure,
            ),
            output_dir=args.output,
            trials_per_gesture=args.trials_per_gesture,
            trial_groups=tuple(args.trial_groups) if args.trial_groups else None,
            roi_size=args.roi_size,
        )
    elif args.command == "record-negative":
        report = record_negative_dataset(
            MediaPipeFaceBackend(args.model),
            CameraConfig(
                index=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                analysis_rate=args.fps,
                preview=True,
                backend=args.capture_backend,
                fourcc=args.fourcc,
                exposure=args.exposure,
            ),
            output_dir=args.output,
            duration_s=args.seconds,
            roi_size=args.roi_size,
        )
    elif args.command == "record-pose":
        report = record_pose_dataset(
            MediaPipeFaceBackend(args.model),
            CameraConfig(
                index=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                analysis_rate=args.fps,
                preview=True,
                backend=args.capture_backend,
                fourcc=args.fourcc,
                exposure=args.exposure,
            ),
            output_dir=args.output,
            seconds_per_pose=args.seconds_per_pose,
            roi_size=args.roi_size,
        )
    elif args.command == "record-pose-sweep":
        report = record_pose_sweep_dataset(
            MediaPipeFaceBackend(args.model),
            CameraConfig(
                index=args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
                analysis_rate=args.fps,
                preview=True,
                backend=args.capture_backend,
                fourcc=args.fourcc,
                exposure=args.exposure,
            ),
            output_dir=args.output,
            seconds_per_step=args.seconds_per_step,
            roi_size=args.roi_size,
        )
    elif args.command == "recover-pose-sweep":
        report = recover_pose_sweep_manifest(
            args.dataset,
            requested_camera=(args.camera_width, args.camera_height, args.camera_fps),
        )
    else:
        report = replay_dataset(
            MediaPipeFaceBackend(args.model),
            dataset_dir=args.dataset,
            export_path=args.export,
            run_sweep=args.sweep,
        )
    print(json.dumps(report, indent=2))
    return 0
