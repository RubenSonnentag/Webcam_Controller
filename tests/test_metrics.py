from webcam_benchmark.metrics import percentile, summarize


def test_percentiles_are_interpolated() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 4.8


def test_summary_empty() -> None:
    assert summarize([])["count"] == 0.0
