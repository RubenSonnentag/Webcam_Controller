from webcam_benchmark.cli import build_parser


def test_record_supports_automatic_advance() -> None:
    args = build_parser().parse_args(
        [
            "record",
            "--model",
            "model.task",
            "--output",
            "dataset",
            "--auto-advance",
            "--group-change-seconds",
            "5",
        ]
    )
    assert args.auto_advance is True
    assert args.group_change_seconds == 5.0


def test_record_mouth_uses_1080p60_mjpeg_defaults() -> None:
    args = build_parser().parse_args(
        [
            "record-mouth",
            "--model",
            "model.task",
            "--output",
            "dataset",
            "--exposure",
            "-6",
        ]
    )
    assert (args.width, args.height, args.fps) == (1920, 1080, 60.0)
    assert args.fourcc == "mjpg"
    assert args.exposure == -6.0
    assert args.trials_per_group == 10


def test_replay_accepts_openseeface_provider() -> None:
    args = build_parser().parse_args(
        [
            "replay",
            "--engine",
            "openseeface",
            "--openseeface-root",
            "OpenSeeFace",
            "--openseeface-model",
            "4",
            "--dataset",
            "dataset",
        ]
    )
    assert args.model is None
    assert args.engine == "openseeface"
    assert args.openseeface_model == 4
