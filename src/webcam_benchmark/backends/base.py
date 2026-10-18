from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import FaceSignals


class FaceBackend(ABC):
    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def analyze(self, image_bgr: object, timestamp_ns: int) -> FaceSignals:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "FaceBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
