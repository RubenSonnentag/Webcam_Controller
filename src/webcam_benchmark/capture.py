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
            if self.config.fourcc != "auto":
                capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*self.config.fourcc.upper()),
                )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.config.index}")
            self.actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
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
