from __future__ import annotations

import threading
from dataclasses import dataclass

from .types import FramePacket


@dataclass(frozen=True, slots=True)
class FrameBufferStats:
    published: int
    overwritten: int
    consumed: int
    first_arrived_ns: int
    last_arrived_ns: int


class LatestFrameBuffer:
    """One-slot buffer: publishing never queues stale frames."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._closed = False
        self._published = 0
        self._overwritten = 0
        self._consumed = 0
        self._first_arrived_ns = 0
        self._last_arrived_ns = 0

    def publish(self, frame: FramePacket) -> None:
        with self._condition:
            if self._closed:
                return
            if self._latest is not None:
                self._overwritten += 1
            self._latest = frame
            self._published += 1
            if self._first_arrived_ns == 0:
                self._first_arrived_ns = frame.arrived_ns
            self._last_arrived_ns = frame.arrived_ns
            self._condition.notify()

    def take(self, timeout_s: float = 0.25) -> FramePacket | None:
        with self._condition:
            if self._latest is None and not self._closed:
                self._condition.wait(timeout_s)
            frame = self._latest
            self._latest = None
            if frame is not None:
                self._consumed += 1
            return frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def stats(self) -> FrameBufferStats:
        with self._condition:
            return FrameBufferStats(
                published=self._published,
                overwritten=self._overwritten,
                consumed=self._consumed,
                first_arrived_ns=self._first_arrived_ns,
                last_arrived_ns=self._last_arrived_ns,
            )
