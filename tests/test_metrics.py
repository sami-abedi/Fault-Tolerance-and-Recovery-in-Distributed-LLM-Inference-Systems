"""
Unit tests for the metrics collection system.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import (
    AggregateStats,
    MetricsSink,
    RequestMetrics,
    compute_aggregate,
)


def make_record(
    request_id: str = "req-1",
    success: bool = True,
    latency_s: float = 1.0,
    recovery_time_s: float = 0.0,
    tokens_recomputed: int = 0,
    total_tokens: int = 50,
    fault_type: str = "none",
    strategy: str = "fail_stop",
) -> RequestMetrics:
    """Helper to create a completed RequestMetrics record."""
    r = RequestMetrics(
        request_id=request_id,
        fault_type=fault_type,
        recovery_strategy=strategy,
        total_tokens_generated=total_tokens,
        tokens_recomputed=tokens_recomputed,
        success=success,
    )
    r.timestamp_start = time.time() - latency_s
    r.timestamp_end = time.time()
    r.timestamp_first_token = r.timestamp_start + latency_s * 0.1
    if recovery_time_s > 0:
        r.timestamp_recovery_start = r.timestamp_start + latency_s * 0.3
        r.timestamp_recovery_end = r.timestamp_recovery_start + recovery_time_s
    return r


class TestRequestMetrics:

    def test_latency_computed_correctly(self):
        r = make_record(latency_s=2.5)
        assert r.latency_s is not None
        assert abs(r.latency_s - 2.5) < 0.1

    def test_ttft_computed(self):
        r = make_record(latency_s=1.0)
        assert r.time_to_first_token_s is not None
        assert r.time_to_first_token_s > 0

    def test_recomputation_fraction(self):
        r = make_record(tokens_recomputed=20, total_tokens=80)
        assert abs(r.recomputation_fraction - 0.25) < 0.01

    def test_recomputation_fraction_zero_total(self):
        r = make_record(tokens_recomputed=0, total_tokens=0)
        assert r.recomputation_fraction == 0.0

    def test_to_dict_contains_derived_fields(self):
        r = make_record(latency_s=1.5, tokens_recomputed=10, total_tokens=50)
        d = r.to_dict()
        assert "latency_s" in d
        assert "time_to_first_token_s" in d
        assert "recomputation_fraction" in d
        assert d["recomputation_fraction"] == pytest.approx(0.2, abs=0.01)


class TestAggregateStats:

    def test_compute_aggregate_success_rate(self):
        records = [make_record(f"req-{i}", success=(i % 2 == 0)) for i in range(10)]
        agg = compute_aggregate(records)
        assert agg.success_rate == pytest.approx(0.5, abs=0.01)
        assert agg.successful_requests == 5
        assert agg.failed_requests == 5

    def test_compute_aggregate_latency_percentiles(self):
        import numpy as np
        latencies = [0.1 * (i + 1) for i in range(20)]
        records = [make_record(f"req-{i}", latency_s=l) for i, l in enumerate(latencies)]
        agg = compute_aggregate(records)
        assert agg.p50_latency_s > 0
        assert agg.p95_latency_s >= agg.p50_latency_s
        assert agg.p99_latency_s >= agg.p95_latency_s

    def test_compute_aggregate_empty_list(self):
        agg = compute_aggregate([])
        assert agg.total_requests == 0
        assert agg.success_rate == 0.0


class TestMetricsSink:

    def test_record_and_export(self, tmp_path):
        sink = MetricsSink(str(tmp_path / "metrics.jsonl"))
        for i in range(5):
            r = make_record(f"req-{i}")
            sink.record(r)

        assert len(sink.records) == 5

        # Verify JSONL file was written
        log_file = tmp_path / "metrics.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "request_id" in obj
            assert "latency_s" in obj

    def test_export_csv(self, tmp_path):
        sink = MetricsSink(str(tmp_path / "metrics.jsonl"))
        for i in range(3):
            sink.record(make_record(f"req-{i}"))

        csv_path = str(tmp_path / "export.csv")
        sink.export_csv(csv_path)

        import csv
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_clear(self, tmp_path):
        sink = MetricsSink(str(tmp_path / "metrics.jsonl"))
        sink.record(make_record("req-1"))
        assert len(sink.records) == 1
        sink.clear()
        assert len(sink.records) == 0
