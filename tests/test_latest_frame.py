from webcam_benchmark.latest_frame import LatestFrameBuffer
from webcam_benchmark.types import FramePacket


def test_latest_frame_overwrites_stale_frame() -> None:
    buffer = LatestFrameBuffer()
    buffer.publish(FramePacket(1, 1, "old"))
    buffer.publish(FramePacket(2, 2, "new"))

    assert buffer.take(0).sequence == 2
    stats = buffer.stats()
    assert stats.published == 2
    assert stats.overwritten == 1
    assert stats.consumed == 1
    assert stats.first_arrived_ns == 1
    assert stats.last_arrived_ns == 2
