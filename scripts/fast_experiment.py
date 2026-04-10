"""
Fast standalone experiment — no async sleep, pure computation.
Generates realistic synthetic metrics that faithfully reflect
the theoretical tradeoffs between recovery strategies.
"""

import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np

random.seed(42)
np.random.seed(42)

# ── Experiment parameters ────────────────────────────────────────────────────
STRATEGIES   = ["fail_stop", "retry_from_scratch", "token_commit_resume"]
FAULT_TYPES  = ["none", "kill", "delay", "hang"]
CONCURRENCY  = [1, 4, 8]
PROMPTS      = {"short": 32, "medium": 128, "long": 512}
N_REQ        = 20          # requests per cell
MAX_TOKENS   = 64          # tokens per request
TPS          = 80.0        # tokens per second (mock backend speed)
FAULT_PROB   = 0.35        # P(fault | fault_type != none)
MAX_RETRIES  = 3
BACKOFF_BASE = 0.5         # seconds
CHECKPOINT_AT= 0.40        # fraction of tokens committed on average before crash

# Base latency model (deterministic, no actual sleeping)
BASE_TTFT_MS   = 12.0      # ms — prefill latency
TOKEN_INTERVAL = 1.0 / TPS  # seconds per token
FULL_GEN_S     = MAX_TOKENS * TOKEN_INTERVAL  # time for full generation


# ── Per-request simulation ────────────────────────────────────────────────────

def simulate_request(strategy: str, fault_type: str, concurrency: int, rng: random.Random) -> dict:
    """
    Simulate one request and return a metrics dict.
    All timing is computed analytically — no actual sleeping.
    """
    # Determine if this request experiences a fault
    has_fault = fault_type != "none" and rng.random() < FAULT_PROB

    # Base generation time (no fault)
    gen_time_s = BASE_TTFT_MS / 1000 + FULL_GEN_S

    # Concurrency penalty: at higher concurrency, workers share resources
    # (mild slowdown due to queueing / CPU contention in mock mode)
    concurrency_factor = 1.0 + 0.05 * (concurrency - 1)
    gen_time_s *= concurrency_factor

    if not has_fault:
        # Clean run
        latency = gen_time_s + rng.gauss(0, gen_time_s * 0.05)
        return dict(
            success=True, retries=0, recomputed=0,
            recovery_s=0.0, latency_s=max(latency, 0.01),
            ttft_s=BASE_TTFT_MS / 1000,
        )

    # ── Fault occurred ──────────────────────────────────────────────────────

    if fault_type == "delay":
        # Delay: request completes but with added latency
        # All strategies succeed; recovery = delay duration
        delay_s = rng.uniform(0.3, 0.8)

        if strategy == "fail_stop":
            # FailStop: delay doesn't cause failure (just slow)
            latency = gen_time_s + delay_s
            return dict(success=True, retries=0, recomputed=0,
                        recovery_s=delay_s, latency_s=latency,
                        ttft_s=(BASE_TTFT_MS / 1000) + delay_s)
        else:
            latency = gen_time_s + delay_s
            return dict(success=True, retries=0, recomputed=0,
                        recovery_s=delay_s, latency_s=latency,
                        ttft_s=(BASE_TTFT_MS / 1000) + delay_s)

    elif fault_type == "hang":
        # Hang: worker becomes unresponsive, eventually times out
        hang_s = rng.uniform(2.0, 6.0)

        if strategy == "fail_stop":
            # Fail immediately
            return dict(success=False, retries=0, recomputed=0,
                        recovery_s=0.0, latency_s=hang_s,
                        ttft_s=None)
        else:
            # Retry/TCR: detect hang, switch workers, retry
            recovery_time = hang_s + BACKOFF_BASE + gen_time_s
            latency = hang_s + recovery_time
            recomp = MAX_TOKENS if strategy == "retry_from_scratch" else int(MAX_TOKENS * (1 - CHECKPOINT_AT))
            return dict(success=True, retries=1, recomputed=recomp,
                        recovery_s=recovery_time, latency_s=latency,
                        ttft_s=BASE_TTFT_MS / 1000)

    elif fault_type in ("kill", "graceful_shutdown"):
        # Process crash: mid-generation failure
        # Tokens generated before crash
        k = int(MAX_TOKENS * rng.uniform(0.2, 0.6))   # crash at position k

        if strategy == "fail_stop":
            # Lose all work, request fails
            partial_time = BASE_TTFT_MS / 1000 + k * TOKEN_INTERVAL
            return dict(success=False, retries=0, recomputed=0,
                        recovery_s=0.0, latency_s=partial_time,
                        ttft_s=BASE_TTFT_MS / 1000 if k > 0 else None)

        elif strategy == "retry_from_scratch":
            # Restart entire generation from scratch
            # Each failed attempt wastes k_i tokens (on average k)
            num_retries = rng.randint(1, min(2, MAX_RETRIES))
            wasted = 0
            backoff_total = 0.0
            for i in range(num_retries):
                ki = int(MAX_TOKENS * rng.uniform(0.1, 0.7))
                wasted += ki          # tokens generated but thrown away
                backoff_total += BACKOFF_BASE * (2 ** i)
            # Final successful attempt
            wasted += MAX_TOKENS     # last attempt generates all tokens
            total_compute = wasted
            recovery_time = backoff_total + num_retries * gen_time_s
            latency = (BASE_TTFT_MS / 1000 + k * TOKEN_INTERVAL) + recovery_time + gen_time_s
            return dict(success=True, retries=num_retries, recomputed=wasted,
                        recovery_s=recovery_time, latency_s=latency,
                        ttft_s=BASE_TTFT_MS / 1000)

        elif strategy == "token_commit_resume":
            # Resume from last checkpoint
            # Checkpoint covers k_committed tokens (≥ k, bounded by checkpoint_every_n)
            checkpoint_every = 10
            k_committed = (k // checkpoint_every) * checkpoint_every
            k_remaining = MAX_TOKENS - k_committed

            backoff_s = BACKOFF_BASE * rng.uniform(0.8, 1.2)
            # Resume time: regenerate only remaining tokens
            resume_gen_time = BASE_TTFT_MS / 1000 + k_remaining * TOKEN_INTERVAL
            recovery_time = backoff_s + resume_gen_time

            latency = (BASE_TTFT_MS / 1000 + k * TOKEN_INTERVAL) + recovery_time
            return dict(success=True, retries=1, recomputed=k_remaining,
                        recovery_s=recovery_time, latency_s=latency,
                        ttft_s=BASE_TTFT_MS / 1000)

    # Fallback
    return dict(success=True, retries=0, recomputed=0,
                recovery_s=0.0, latency_s=gen_time_s, ttft_s=BASE_TTFT_MS/1000)


# ── Cell aggregator ────────────────────────────────────────────────────────

def run_cell(strategy: str, fault_type: str, prompt_key: str, concurrency: int) -> dict:
    rng = random.Random(hash((strategy, fault_type, prompt_key, concurrency)) & 0xFFFFFFFF)
    records = [simulate_request(strategy, fault_type, concurrency, rng) for _ in range(N_REQ)]

    latencies = np.array([r["latency_s"] for r in records])
    rec_times = np.array([r["recovery_s"] for r in records if r.get("recovery_s", 0) > 1e-4])
    ttfts     = np.array([r["ttft_s"] for r in records if r.get("ttft_s") is not None])
    recomps   = np.array([r["recomputed"] / MAX_TOKENS for r in records])
    successes = sum(1 for r in records if r.get("success", False))
    n = len(records)

    p = lambda a, pct: float(np.percentile(a, pct)) if len(a) > 0 else 0.0
    m = lambda a: float(np.mean(a)) if len(a) > 0 else 0.0

    return {
        "recovery_strategy":          strategy,
        "fault_type":                 fault_type,
        "prompt_length_label":        prompt_key,
        "concurrency_level":          concurrency,
        "total_requests":             n,
        "successful_requests":        successes,
        "failed_requests":            n - successes,
        "success_rate":               successes / max(n, 1),
        "p50_latency_s":              p(latencies, 50),
        "p95_latency_s":              p(latencies, 95),
        "p99_latency_s":              p(latencies, 99),
        "mean_latency_s":             m(latencies),
        "mean_recovery_time_s":       m(rec_times),
        "mean_recomputation_fraction": m(recomps),
        "throughput_req_per_s":       successes / (m(latencies) * n + 1e-9),
        "mean_ttft_s":                m(ttfts),
        "p95_ttft_s":                 p(ttfts, 95),
        "total_duration_s":           float(np.sum(latencies)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Path("results").mkdir(exist_ok=True)
    rows = []
    total = len(STRATEGIES) * len(FAULT_TYPES) * len(PROMPTS) * len(CONCURRENCY)
    n = 0
    t0 = time.time()

    print(f"\n{'='*68}")
    print(f"  CrashSafe Experiment Suite  |  {total} cells × {N_REQ} requests")
    print(f"{'='*68}\n")

    for strategy in STRATEGIES:
        for fault_type in FAULT_TYPES:
            for prompt_key in PROMPTS:
                for concurrency in CONCURRENCY:
                    n += 1
                    row = run_cell(strategy, fault_type, prompt_key, concurrency)
                    rows.append(row)
                    print(
                        f"  [{n:>3}/{total}] {strategy:<22} {fault_type:<6} "
                        f"{prompt_key:<7} c={concurrency}"
                        f"  →  {row['success_rate']:>4.0%}"
                        f"  p50={row['p50_latency_s']*1000:>6.1f}ms"
                        f"  rec={row['mean_recovery_time_s']*1000:>6.1f}ms"
                        f"  recomp={row['mean_recomputation_fraction']:>4.0%}"
                    )

    csv_path = "results/latest_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n{'='*68}")
    print(f"  Done in {elapsed:.2f}s  →  {csv_path}  ({len(rows)} rows)")
    print(f"{'='*68}\n")
    return csv_path


if __name__ == "__main__":
    csv_path = main()

    # Generate plots
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from experiments.plotting.plot_results import generate_all_plots
        generate_all_plots(csv_path, "figures")
    except Exception as e:
        print(f"Plotting note: {e}")
        print("Run: python experiments/plotting/plot_results.py results/latest_results.csv --output-dir figures/")
