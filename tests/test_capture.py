from webcam_benchmark.capture import _configure_capture, _decode_fourcc
from webcam_benchmark.config import CameraConfig


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_FOURCC = 4
    CAP_PROP_BUFFERSIZE = 5
    CAP_PROP_AUTO_EXPOSURE = 6
    CAP_PROP_EXPOSURE = 7

    @staticmethod
    def VideoWriter_fourcc(*characters: str) -> int:
        return sum(ord(character) << (8 * index) for index, character in enumerate(characters))


class FakeCapture:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def set(self, key: int, value: float) -> bool:
        self.calls.append((key, value))
        return True


def test_uvc_format_is_set_after_resolution_and_fps() -> None:
    capture = FakeCapture()
    config = CameraConfig(
        width=1920,
        height=1080,
        fps=60.0,
        fourcc="mjpg",
        exposure=-6.0,
    )
    _configure_capture(capture, config, FakeCv2)
    assert [key for key, _ in capture.calls] == [1, 2, 3, 4, 5, 6, 7]
    assert capture.calls[3][1] == FakeCv2.VideoWriter_fourcc(*"MJPG")


def test_fourcc_decode_round_trip() -> None:
    value = FakeCv2.VideoWriter_fourcc(*"MJPG")
    assert _decode_fourcc(float(value)) == "MJPG"
