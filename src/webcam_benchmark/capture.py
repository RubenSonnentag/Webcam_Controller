from __future__ import annotations

import threading
import time

from .config import CameraConfig
from .latest_frame import LatestFrameBuffer
from .types import FramePacket


class CameraCapture:
    def __init__(self, config: CameraConfig, target: LatestFrameBuffer) -> None:
        self.config = config
        self.target = target
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.actual_fourcc = ""
        self.actual_exposure = 0.0
        self.error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.target.close()

    def _run(self) -> None:
        import cv2

        backends = {
            "auto": cv2.CAP_ANY,
            "msmf": cv2.CAP_MSMF,
            "dshow": cv2.CAP_DSHOW,
        }
        capture = cv2.VideoCapture(self.config.index, backends[self.config.backend])
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.config.index}")
            _configure_capture(capture, self.config, cv2)
            self.actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.actual_fourcc = _decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
            self.actual_exposure = float(capture.get(cv2.CAP_PROP_EXPOSURE))
            sequence = 0
            while not self._stop.is_set():
                ok, image = capture.read()
                arrived_ns = time.perf_counter_ns()
                if not ok:
                    raise RuntimeError("Camera frame acquisition failed")
                self.target.publish(FramePacket(sequence, arrived_ns, image))
                sequence += 1
        except Exception as exc:  # surfaced to the coordinator
            self.error = exc
        finally:
            capture.release()
            self.target.close()


def _configure_capture(capture: object, config: CameraConfig, cv2: object) -> None:
    # Some UVC/DirectShow drivers only retain MJPG when FourCC is set after
    # resolution and FPS. Reversing this order silently fell back to YUY2.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)  # type: ignore[attr-defined]
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)  # type: ignore[attr-defined]
    capture.set(cv2.CAP_PROP_FPS, config.fps)  # type: ignore[attr-defined]
    if config.fourcc != "auto":
        capture.set(  # type: ignore[attr-defined]
            cv2.CAP_PROP_FOURCC,  # type: ignore[attr-defined]
            cv2.VideoWriter_fourcc(*config.fourcc.upper()),  # type: ignore[attr-defined]
        )
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # type: ignore[attr-defined]
    if config.exposure is not None:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # type: ignore[attr-defined]
        capture.set(cv2.CAP_PROP_EXPOSURE, config.exposure)  # type: ignore[attr-defined]


def _decode_fourcc(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4)).rstrip("\x00")
