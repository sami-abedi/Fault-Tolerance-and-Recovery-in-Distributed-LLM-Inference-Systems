"""
Per-request metrics collection and aggregate statistics.

Captures latency, TTFT, recovery overhead, token recomputation,
and success/failure per request. Supports JSONL logging and CSV export.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-request record
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    """Metrics captured for a single inference request."""

    request_id: str
    timestamp_start: float = field(default_factory=time.time)

    # Timing fields (filled in progressively)
    timestamp_first_token: Optional[float] = None
    timestamp_end: Optional[float] = None
    timestamp_recovery_start: Optional[float] = None
    timestamp_recovery_end: Optional[float] = None

    # Fault / recovery info
    fault_type: str = "none"
    recovery_strategy: str = "none"
    num_retries: int = 0
    tokens_committed_before_failure: int = 0
    tokens_recomputed: int = 0
    total_tokens_generated: int = 0

    # Prompt metadata
    prompt_length: int = 0
    concurrency_level: int = 1

    # Outcome
    success: bool = False
    error_message: Optional[str] = None

    # ----------------------------------------------------------------
    # Derived properties
    # ----------------------------------------------------------------

    @property
    def latency_s(self) -> Optional[float]:
        """End-to-end latency in seconds."""
        if self.timestamp_end is not None:
            return self.timestamp_end - self.timestamp_start
        return None

    @property
    def time_to_first_token_s(self) -> Optional[float]:
        """Time from request start to first token emitted."""
        if self.timestamp_first_token is not None:
            return self.timestamp_first_token - self.timestamp_start
        return None

    @property
    def recovery_time_s(self) -> Optional[float]:
        """Time spent in recovery (retry or checkpoint restore)."""
        if (
            self.timestamp_recovery_start is not None
            and self.timestamp_recovery_end is not None
        ):
            return self.timestamp_recovery_end - self.timestamp_recovery_start
        return None

    @property
    def recomputation_fraction(self) -> float:
        """Fraction of tokens that had to be recomputed after failure."""
        if self.total_tokens_generated == 0:
            return 0.0
        return self.tokens_recomputed / self.total_tokens_generated

    def to_dict(self) -> Dict:
        """Serialize to flat dictionary (for JSONL / CSV)."""
        d = asdict(self)
        d["latency_s"] = self.latency_s
        d["time_to_first_token_s"] = self.time_to_first_token_s
        d["recovery_time_s"] = self.recovery_time_s
        d["recomputation_fraction"] = self.recomputation_fraction
        return d


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


@dataclass
class AggregateStats:
    """Aggregate statistics over a set of requests."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency percentiles (seconds)
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    mean_latency_s: float = 0.0

    # TTFT
    mean_ttft_s: float = 0.0
    p95_ttft_s: float = 0.0

    # Recovery
    mean_recovery_time_s: float = 0.0
    mean_recomputation_fraction: float = 0.0

    # Throughput
    throughput_req_per_s: float = 0.0
    total_duration_s: float = 0.0

    success_rate: float = 0.0

    # Experiment labels
    fault_type: str = "none"
    recovery_strategy: str = "none"
    concurrency_level: int = 1
    prompt_length_label: str = "short"

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_aggregate(
    records: List[RequestMetrics],
    fault_type: str = "none",
    recovery_strategy: str = "none",
    concurrency_level: int = 1,
    prompt_length_label: str = "short",
) -> AggregateStats:
    """
    Compute aggregate statistics from a list of RequestMetrics.

    Args:
        records: List of completed request metrics.
        fault_type: Fault injection type label.
        recovery_strategy: Recovery strategy label.
        concurrency_level: Number of concurrent workers.
        prompt_length_label: Human-readable prompt length bucket.

    Returns:
        AggregateStats with computed percentiles and rates.
    """
    stats = AggregateStats(
        fault_type=fault_type,
        recovery_strategy=recovery_strategy,
        concurrency_level=concurrency_level,
        prompt_length_label=prompt_length_label,
    )

    if not records:
        return stats

    stats.total_requests = len(records)
    stats.successful_requests = sum(1 for r in records if r.success)
    stats.failed_requests = stats.total_requests - stats.successful_requests
    stats.success_rate = stats.successful_requests / stats.total_requests

    latencies = [r.latency_s for r in records if r.latency_s is not None]
    if latencies:
        arr = np.array(latencies)
        stats.p50_latency_s = float(np.percentile(arr, 50))
        stats.p95_latency_s = float(np.percentile(arr, 95))
        stats.p99_latency_s = float(np.percentile(arr, 99))
        stats.mean_latency_s = float(np.mean(arr))

    ttfts = [
        r.time_to_first_token_s
        for r in records
        if r.time_to_first_token_s is not None
    ]
    if ttfts:
        stats.mean_ttft_s = float(np.mean(ttfts))
        stats.p95_ttft_s = float(np.percentile(ttfts, 95))

    recovery_times = [
        r.recovery_time_s for r in records if r.recovery_time_s is not None
    ]
    if recovery_times:
        stats.mean_recovery_time_s = float(np.mean(recovery_times))

    stats.mean_recomputation_fraction = float(
        np.mean([r.recomputation_fraction for r in records])
    )

    # Throughput: total successful requests / wall-clock window
    if records:
        t_start = min(r.timestamp_start for r in records)
        t_end = max(
            r.timestamp_end
            for r in records
            if r.timestamp_end is not None
        ) if any(r.timestamp_end for r in records) else time.time()
        stats.total_duration_s = t_end - t_start
        if stats.total_duration_s > 0:
            stats.throughput_req_per_s = (
                stats.successful_requests / stats.total_duration_s
            )

    return stats


# ---------------------------------------------------------------------------
# Metrics sink (JSONL writer)
# ---------------------------------------------------------------------------


class MetricsSink:
    """
    Thread-safe, async-friendly JSONL metrics logger.

    Appends one JSON object per line for each completed request.
    """

    def __init__(self, log_path: str) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[RequestMetrics] = []

    def record(self, metrics: RequestMetrics) -> None:
        """Append a request metric record to the in-memory list and JSONL file."""
        self._records.append(metrics)
        with open(self._path, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")

    def export_csv(self, csv_path: str) -> None:
        """Export all recorded metrics to a CSV file."""
        if not self._records:
            return
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in self._records]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    @property
    def records(self) -> List[RequestMetrics]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
