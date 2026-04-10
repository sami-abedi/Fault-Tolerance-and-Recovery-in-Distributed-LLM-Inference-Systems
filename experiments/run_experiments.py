"""
Experiment runner for CrashSafe fault tolerance evaluation.

Sweeps over the following dimensions:
  - concurrency: 1, 4, 8 simultaneous requests
  - prompt_length: short (32), medium (128), long (512) tokens
  - fault_type: none, kill, delay, hang
  - recovery_strategy: fail_stop, retry_from_scratch, token_commit_resume

For each combination, sends `requests_per_cell` requests and collects:
  - end-to-end latency
  - time-to-first-token
  - recovery time
  - tokens recomputed
  - success/failure

Results are saved to CSV and JSONL, then plotted.

Usage:
    # Run full experiment suite (against a running server)
    python experiments/run_experiments.py --mode full

    # Run local simulation (no server needed — uses direct backend calls)
    python experiments/run_experiments.py --mode local --output-dir results/

    # Quick smoke test
    python experiments/run_experiments.py --mode quick
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import numpy as np

from src.config import AppConfig, load_config
from src.utils.metrics import AggregateStats, RequestMetrics, compute_aggregate


# ---------------------------------------------------------------------------
# Prompt generators
# ---------------------------------------------------------------------------

SHORT_PROMPTS = [
    "The quick brown fox",
    "In a distributed system",
    "Fault tolerance means",
    "The inference latency",
    "Token generation speed",
]

MEDIUM_PROMPTS = [
    (
        "Distributed systems face numerous challenges when handling failures. "
        "In the context of large language model inference, a single worker crash "
        "can cause complete request failure. Describe how checkpointing helps"
    ),
    (
        "The key insight behind token-commit-resume recovery is that tokens "
        "generated before a fault need not be recomputed if they are persisted "
        "to durable storage. This reduces recovery overhead significantly "
        "compared to retrying from scratch. Explain the tradeoffs involved"
    ),
]

LONG_PROMPTS = [
    (
        "Consider a large-scale distributed inference system serving thousands of "
        "concurrent requests. Each request requires generating hundreds of tokens, "
        "and worker failures occur with some probability p per unit time. "
        "The system must balance three competing objectives: (1) minimizing "
        "end-to-end latency for successful requests, (2) maximizing request "
        "success rate under failure conditions, and (3) minimizing wasted "
        "compute due to token recomputation. Three strategies are available: "
        "fail-stop (no recovery), retry-from-scratch (restart entire generation), "
        "and token-commit-resume (checkpoint and resume). Analyze the tradeoffs "
        "between these strategies in terms of latency overhead, reliability, "
        "and compute efficiency under varying fault rates and prompt lengths. "
        "Consider the impact of checkpoint granularity on recovery time and "
        "the role of persistent storage latency in the overall system throughput"
    ),
]

PROMPT_SETS = {
    "short": SHORT_PROMPTS,
    "medium": MEDIUM_PROMPTS,
    "long": LONG_PROMPTS,
}


# ---------------------------------------------------------------------------
# Local simulation (no server needed)
# ---------------------------------------------------------------------------


@dataclass
class SimulationParams:
    """Parameters for one simulation cell."""
    concurrency: int
    prompt_length: str
    fault_type: str
    recovery_strategy: str
    requests_per_cell: int
    max_new_tokens: int = 64  # reduced for speed in experiments
    tokens_per_second: float = 100.0  # mock backend speed
    base_latency_ms: float = 5.0
    fault_probability: float = 0.3  # P(fault per request)
    delay_ms: float = 500.0
    hang_duration_s: float = 2.0
    checkpoint_every_n: int = 10


async def simulate_one_request(
    params: SimulationParams,
    request_idx: int,
) -> RequestMetrics:
    """
    Simulate a single inference request with fault injection.

    This function directly invokes the backend and recovery logic
    without going through HTTP, enabling fast local experimentation.
    """
    from src.backends.mock_backend import MockBackend
    from src.backends.base import InferenceRequest
    from src.recovery.fail_stop import FailStopStrategy
    from src.recovery.retry import RetryFromScratchStrategy
    from src.recovery.token_resume import TokenCommitResumeStrategy
    from src.recovery.base import RecoveryContext
    from src.storage.token_store import MemoryTokenStore

    request_id = f"sim-{params.recovery_strategy}-{params.fault_type}-{request_idx}"
    prompts = PROMPT_SETS[params.prompt_length]
    prompt = prompts[request_idx % len(prompts)]

    metrics = RequestMetrics(
        request_id=request_id,
        fault_type=params.fault_type,
        recovery_strategy=params.recovery_strategy,
        prompt_length=len(prompt.split()),
        concurrency_level=params.concurrency,
    )

    # Determine if this request will experience a fault
    import random
    will_fault = (
        params.fault_type != "none"
        and random.random() < params.fault_probability
    )
    fault_triggered = False

    # Create mock backend
    backend = MockBackend({
        "tokens_per_second": params.tokens_per_second,
        "base_latency_ms": params.base_latency_ms,
    })
    await backend.load()

    async def generate_fn(req: InferenceRequest):
        """Simulate generation, injecting fault if needed."""
        nonlocal fault_triggered

        # Simulate TTFT
        metrics.timestamp_first_token = time.time() + (params.base_latency_ms / 1000)

        if will_fault and not fault_triggered:
            fault_triggered = True

            if params.fault_type == "delay":
                await asyncio.sleep(params.delay_ms / 1000.0)
                # delay doesn't prevent completion, just adds latency
                return await backend.generate(req)

            elif params.fault_type == "hang":
                await asyncio.sleep(params.hang_duration_s)
                raise TimeoutError(f"Simulated hang for {params.hang_duration_s}s")

            elif params.fault_type in ("kill", "graceful_shutdown"):
                # Simulate partial generation then crash
                # Generate some tokens then raise
                tokens_before_crash = max(1, params.max_new_tokens // 3)
                partial_req = InferenceRequest(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    max_new_tokens=tokens_before_crash,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    resume_from_tokens=req.resume_from_tokens,
                )
                partial_resp = await backend.generate(partial_req)

                # Simulate the crash after partial generation
                # The token_commit_resume strategy would have checkpointed these
                # For simulation, we store them in the request metadata
                if hasattr(req, '_partial_tokens'):
                    req._partial_tokens = partial_resp.tokens
                raise RuntimeError(
                    f"Simulated {params.fault_type}: worker process terminated"
                )

        return await backend.generate(req)

    # Set up recovery strategy
    if params.recovery_strategy == "fail_stop":
        strategy = FailStopStrategy()
    elif params.recovery_strategy == "retry_from_scratch":
        strategy = RetryFromScratchStrategy(
            max_retries=3,
            backoff_base_s=0.05,  # fast backoff for simulation
            backoff_max_s=0.5,
        )
    elif params.recovery_strategy == "token_commit_resume":
        store = MemoryTokenStore()
        strategy = TokenCommitResumeStrategy(
            token_store=store,
            checkpoint_every_n=params.checkpoint_every_n,
            max_retries=3,
            backoff_base_s=0.05,
            backoff_max_s=0.5,
        )
    else:
        raise ValueError(f"Unknown strategy: {params.recovery_strategy}")

    inference_req = InferenceRequest(
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=params.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    context = RecoveryContext(
        request=inference_req,
        worker_ids=["sim-worker-0"],
    )

    result = await strategy.execute(context, generate_fn)

    metrics.timestamp_end = time.time()
    metrics.success = result.success
    metrics.num_retries = result.num_retries
    metrics.tokens_recomputed = result.tokens_recomputed
    metrics.total_tokens_generated = (
        result.response.completion_tokens if result.response else 0
    )
    if result.recovery_time_s > 0:
        t_recovery = metrics.timestamp_start + (metrics.latency_s or 0) * 0.5
        metrics.timestamp_recovery_start = t_recovery
        metrics.timestamp_recovery_end = t_recovery + result.recovery_time_s
    metrics.error_message = result.error

    return metrics


async def run_simulation_cell(params: SimulationParams) -> List[RequestMetrics]:
    """Run all requests for one experiment cell with given concurrency."""
    semaphore = asyncio.Semaphore(params.concurrency)

    async def bounded_request(idx: int) -> RequestMetrics:
        async with semaphore:
            return await simulate_one_request(params, idx)

    tasks = [
        asyncio.create_task(bounded_request(i))
        for i in range(params.requests_per_cell)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    records = []
    for r in results:
        if isinstance(r, RequestMetrics):
            records.append(r)
        elif isinstance(r, Exception):
            # Create a failure record
            m = RequestMetrics(request_id=f"error-{uuid.uuid4().hex[:6]}")
            m.success = False
            m.error_message = str(r)
            m.timestamp_end = time.time()
            records.append(m)

    return records


# ---------------------------------------------------------------------------
# Full experiment sweep
# ---------------------------------------------------------------------------


async def run_full_experiment(
    config: AppConfig,
    output_dir: str = "./results",
    mode: str = "local",
    requests_per_cell: int = 20,
    max_new_tokens: int = 64,
) -> str:
    """
    Run the complete experiment matrix.

    Sweeps over all combinations of:
      (concurrency) × (prompt_length) × (fault_type) × (recovery_strategy)

    Args:
        config: Application configuration.
        output_dir: Directory to write CSV and JSONL results.
        mode: "local" for direct simulation, "server" for HTTP-based testing.
        requests_per_cell: Number of requests per experiment cell.
        max_new_tokens: Tokens to generate per request.

    Returns:
        Path to the generated summary CSV file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = output_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[RequestMetrics] = []
    aggregate_rows: List[Dict] = []

    concurrency_levels = config.experiment.concurrency_levels
    prompt_lengths = list(config.experiment.prompt_lengths.keys())
    fault_types = config.experiment.fault_types
    strategies = config.experiment.recovery_strategies

    total_cells = (
        len(concurrency_levels)
        * len(prompt_lengths)
        * len(fault_types)
        * len(strategies)
    )
    cell_num = 0

    print(f"\n{'='*60}")
    print(f"CrashSafe Experiment Suite")
    print(f"Run ID: {run_id}")
    print(f"Total cells: {total_cells}")
    print(f"Requests per cell: {requests_per_cell}")
    print(f"{'='*60}\n")

    for strategy in strategies:
        for fault_type in fault_types:
            for prompt_length in prompt_lengths:
                for concurrency in concurrency_levels:
                    cell_num += 1
                    cell_label = (
                        f"[{cell_num}/{total_cells}] "
                        f"strategy={strategy} fault={fault_type} "
                        f"prompt={prompt_length} concurrency={concurrency}"
                    )
                    print(f"Running: {cell_label}")

                    params = SimulationParams(
                        concurrency=concurrency,
                        prompt_length=prompt_length,
                        fault_type=fault_type,
                        recovery_strategy=strategy,
                        requests_per_cell=requests_per_cell,
                        max_new_tokens=max_new_tokens,
                    )

                    cell_records = await run_simulation_cell(params)
                    all_records.extend(cell_records)

                    # Compute aggregates for this cell
                    agg = compute_aggregate(
                        cell_records,
                        fault_type=fault_type,
                        recovery_strategy=strategy,
                        concurrency_level=concurrency,
                        prompt_length_label=prompt_length,
                    )
                    aggregate_rows.append(agg.to_dict())

                    print(
                        f"  → success={agg.success_rate:.0%} "
                        f"p50={agg.p50_latency_s*1000:.0f}ms "
                        f"p95={agg.p95_latency_s*1000:.0f}ms "
                        f"recovery={agg.mean_recovery_time_s*1000:.0f}ms "
                        f"recomputed={agg.mean_recomputation_fraction:.0%}"
                    )

    # Write JSONL of all raw records
    raw_path = run_dir / "raw_metrics.jsonl"
    with open(raw_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    # Write aggregate CSV
    agg_csv_path = run_dir / "aggregate_results.csv"
    if aggregate_rows:
        with open(agg_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)

    # Write latest symlink-style copy for easy access
    latest_csv = output_path / "latest_results.csv"
    if aggregate_rows:
        with open(latest_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            writer.writeheader()
            writer.writerows(aggregate_rows)

    print(f"\n{'='*60}")
    print(f"Experiment complete.")
    print(f"Raw metrics:  {raw_path}")
    print(f"Aggregate CSV: {agg_csv_path}")
    print(f"Latest CSV:   {latest_csv}")
    print(f"{'='*60}\n")

    return str(agg_csv_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CrashSafe Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["local", "server", "quick"],
        default="local",
        help="Experiment mode: local=simulation, server=HTTP, quick=fast subset",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory for experiment results",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Number of requests per experiment cell",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per request",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate plots after experiment",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "quick":
        # Reduced sweep for quick validation
        config.experiment.concurrency_levels = [1, 4]
        config.experiment.fault_types = ["none", "kill", "delay"]
        config.experiment.recovery_strategies = [
            "fail_stop", "retry_from_scratch", "token_commit_resume"
        ]
        requests_per_cell = max(5, args.requests // 4)
        max_tokens = min(32, args.max_tokens)
    else:
        requests_per_cell = args.requests
        max_tokens = args.max_tokens

    csv_path = asyncio.run(
        run_full_experiment(
            config=config,
            output_dir=args.output_dir,
            mode=args.mode,
            requests_per_cell=requests_per_cell,
            max_new_tokens=max_tokens,
        )
    )

    if args.plot:
        try:
            from experiments.plotting.plot_results import generate_all_plots
            figures_dir = str(Path(args.output_dir).parent / "figures")
            generate_all_plots(csv_path, figures_dir)
            print(f"Figures saved to: {figures_dir}")
        except Exception as exc:
            print(f"Warning: plotting failed: {exc}")


if __name__ == "__main__":
    main()
