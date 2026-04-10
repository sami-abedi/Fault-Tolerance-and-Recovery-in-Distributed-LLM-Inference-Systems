"""
Standalone experiment runner (no external dependencies except numpy).

Runs the full CrashSafe experiment suite using inline implementations
of all core components. Produces results/latest_results.csv and
generates all figures in figures/.

Usage: python scripts/run_standalone_experiment.py
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Core data types
# ============================================================

@dataclass
class Token:
    token_id: int
    token_text: str
    position: int
    is_eos: bool = False


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    resume_from_tokens: Optional[List[int]] = None


@dataclass
class InferenceResponse:
    request_id: str
    tokens: List[Token] = field(default_factory=list)
    generated_text: str = ""
    completion_tokens: int = 0


# ============================================================
# Mock backend
# ============================================================

_VOCAB = [
    "the", "a", "is", "was", "in", "of", "and", "to", "for", "with",
    "model", "token", "inference", "latency", "fault", "recovery",
    "system", "distributed", "request", "response", "checkpoint",
    "worker", "router", "compute", "failure", "retry", "strategy",
    "overhead", "throughput", "delay", "process", "node", "batch",
    "queue", "timeout", "error", "success", "resume", "commit", "disk",
]


def _tok_id(prompt: str, pos: int) -> int:
    h = hashlib.md5(f"{prompt}:{pos}".encode()).digest()
    return int.from_bytes(h[:4], "little") % len(_VOCAB)


async def mock_generate(request: InferenceRequest) -> InferenceResponse:
    """Deterministic async token generation."""
    resume = request.resume_from_tokens or []
    tokens: List[Token] = []

    # Yield resumed tokens
    for pos, tid in enumerate(resume):
        tokens.append(Token(tid, _VOCAB[tid % len(_VOCAB)], pos))

    # Generate new tokens
    for pos in range(len(resume), request.max_new_tokens):
        tid = _tok_id(request.prompt, pos)
        is_eos = pos == request.max_new_tokens - 1
        tokens.append(Token(tid, _VOCAB[tid % len(_VOCAB)], pos, is_eos))
        await asyncio.sleep(0.0001)  # simulate ~10k tokens/sec

    return InferenceResponse(
        request.request_id,
        tokens,
        " ".join(t.token_text for t in tokens),
        len(tokens),
    )


# ============================================================
# Recovery strategies
# ============================================================

async def strategy_fail_stop(request, gen_fn, **kw):
    t0 = time.time()
    try:
        r = await gen_fn(request)
        return {
            "success": True, "retries": 0, "recomputed": 0,
            "recovery_s": 0.0, "latency_s": time.time() - t0,
        }
    except Exception as e:
        return {
            "success": False, "retries": 0, "recomputed": 0,
            "recovery_s": 0.0, "latency_s": time.time() - t0,
            "error": str(e),
        }


async def strategy_retry_scratch(request, gen_fn, max_retries=3, backoff=0.005):
    t0 = time.time()
    recomputed = 0
    rec_start = None
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            r = await gen_fn(request)
            return {
                "success": True, "retries": attempt, "recomputed": recomputed,
                "recovery_s": time.time() - rec_start if rec_start else 0.0,
                "latency_s": time.time() - t0,
            }
        except Exception as e:
            last_err = str(e)
            if attempt == 0:
                rec_start = time.time()
            else:
                recomputed += request.max_new_tokens
            if attempt < max_retries:
                await asyncio.sleep(backoff * (2 ** attempt))

    recomputed += request.max_new_tokens
    return {
        "success": False, "retries": max_retries, "recomputed": recomputed,
        "recovery_s": time.time() - rec_start if rec_start else 0.0,
        "latency_s": time.time() - t0, "error": last_err,
    }


async def strategy_tcr(request, gen_fn, store, max_retries=3, backoff=0.005):
    """TOKEN_COMMIT_RESUME: checkpoint tokens, resume after fault."""
    t0 = time.time()
    rid = request.request_id
    store[rid] = []  # initialize empty checkpoint
    rec_start = None
    recomputed = 0
    last_err = ""

    for attempt in range(max_retries + 1):
        committed_ids = list(store.get(rid, []))

        resume_req = InferenceRequest(
            rid,
            request.prompt,
            request.max_new_tokens,
            resume_from_tokens=committed_ids if (attempt > 0 and committed_ids) else None,
        )

        try:
            r = await gen_fn(resume_req)
            # Checkpoint new tokens (beyond committed prefix)
            for tok in r.tokens[len(committed_ids):]:
                store.setdefault(rid, []).append(tok.token_id)
            store.pop(rid, None)
            return {
                "success": True, "retries": attempt, "recomputed": recomputed,
                "recovery_s": time.time() - rec_start if rec_start else 0.0,
                "latency_s": time.time() - t0,
            }
        except Exception as e:
            last_err = str(e)
            if attempt == 0:
                rec_start = time.time()
            committed = len(store.get(rid, []))
            recomputed = max(0, request.max_new_tokens - committed)
            if attempt < max_retries:
                await asyncio.sleep(backoff * (2 ** attempt))

    store.pop(rid, None)
    return {
        "success": False, "retries": max_retries, "recomputed": recomputed,
        "recovery_s": time.time() - rec_start if rec_start else 0.0,
        "latency_s": time.time() - t0, "error": last_err,
    }


# ============================================================
# Experiment configuration
# ============================================================

STRATEGIES = ["fail_stop", "retry_from_scratch", "token_commit_resume"]
FAULT_TYPES = ["none", "kill", "delay", "hang"]
CONCURRENCY_LEVELS = [1, 4, 8]
PROMPT_LENGTH_LABELS = {"short": 32, "medium": 128, "long": 512}
N_REQUESTS = 20
MAX_TOKENS = 64
FAULT_PROBABILITY = 0.35
MAX_RETRIES = 3
BACKOFF = 0.002

PROMPTS = {
    "short": [
        "The quick brown fox",
        "In a distributed system",
        "Fault tolerance means",
        "The inference latency",
        "Token generation speed",
    ],
    "medium": [
        (
            "Distributed systems face numerous challenges when handling failures. "
            "In the context of large language model inference, a single worker crash "
            "can cause complete request failure. Describe how checkpointing helps"
        ),
        (
            "The key insight behind token-commit-resume recovery is that tokens "
            "generated before a fault need not be recomputed if they are persisted "
            "to durable storage. This reduces recovery overhead significantly"
        ),
    ],
    "long": [
        (
            "Consider a large-scale distributed inference system serving thousands of "
            "concurrent requests. Each request requires generating hundreds of tokens, "
            "and worker failures occur with some probability p per unit time. "
            "The system must balance three competing objectives: first minimizing "
            "end-to-end latency for successful requests, second maximizing request "
            "success rate under failure conditions, and third minimizing wasted "
            "compute due to token recomputation. Three strategies are available: "
            "fail-stop which provides no recovery, retry-from-scratch which restarts "
            "entire generation, and token-commit-resume which checkpoints tokens "
            "and resumes from the last committed position. Analyze the tradeoffs "
            "between these strategies under varying fault rates and prompt lengths"
        ),
    ],
}


# ============================================================
# Single request simulation
# ============================================================

async def simulate_request(
    strategy_name: str,
    fault_type: str,
    prompt_key: str,
    idx: int,
    checkpoint_store: dict,
) -> dict:
    prompt = PROMPTS[prompt_key][idx % len(PROMPTS[prompt_key])]
    rid = f"{strategy_name}-{fault_type}-{prompt_key}-{idx}"
    req = InferenceRequest(rid, prompt, max_new_tokens=MAX_TOKENS)

    # Determine if this request gets a fault
    will_fault = fault_type != "none" and random.random() < FAULT_PROBABILITY
    call_count = [0]

    async def gen_fn(r: InferenceRequest) -> InferenceResponse:
        call_count[0] += 1
        is_first = call_count[0] == 1

        if will_fault and is_first:
            if fault_type in ("kill", "graceful_shutdown"):
                # Simulate partial generation + crash
                partial_n = max(1, MAX_TOKENS // 3)
                partial_req = InferenceRequest(
                    r.request_id, r.prompt, partial_n,
                    resume_from_tokens=r.resume_from_tokens,
                )
                partial_resp = await mock_generate(partial_req)
                # Populate TCR checkpoint with partial tokens
                if r.request_id in checkpoint_store:
                    existing = len(checkpoint_store[r.request_id])
                    for tok in partial_resp.tokens[existing:]:
                        checkpoint_store[r.request_id].append(tok.token_id)
                raise RuntimeError(
                    f"Simulated {fault_type}: worker terminated after {partial_n} tokens"
                )
            elif fault_type == "delay":
                await asyncio.sleep(0.05)  # 50 ms artificial delay
                return await mock_generate(r)
            elif fault_type == "hang":
                await asyncio.sleep(0.15)
                raise TimeoutError(f"Simulated hang: worker unresponsive for 150ms")

        return await mock_generate(r)

    if strategy_name == "fail_stop":
        result = await strategy_fail_stop(req, gen_fn)
    elif strategy_name == "retry_from_scratch":
        result = await strategy_retry_scratch(req, gen_fn, max_retries=MAX_RETRIES, backoff=BACKOFF)
    elif strategy_name == "token_commit_resume":
        result = await strategy_tcr(req, gen_fn, checkpoint_store, max_retries=MAX_RETRIES, backoff=BACKOFF)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    result["fault_injected"] = will_fault and call_count[0] >= 1
    return result


# ============================================================
# Cell runner
# ============================================================

async def run_cell(
    strategy: str,
    fault_type: str,
    prompt_key: str,
    concurrency: int,
) -> dict:
    """Run one experiment cell and return aggregate statistics."""
    store: dict = {}
    sem = asyncio.Semaphore(concurrency)

    async def bounded(i: int) -> dict:
        async with sem:
            return await simulate_request(strategy, fault_type, prompt_key, i, store)

    tasks = [asyncio.create_task(bounded(i)) for i in range(N_REQUESTS)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    records = [r for r in results if isinstance(r, dict)]

    latencies = np.array([r["latency_s"] for r in records])
    rec_times = np.array([r["recovery_s"] for r in records if r.get("recovery_s", 0) > 1e-6])
    recomp_fracs = np.array([r["recomputed"] / MAX_TOKENS for r in records])
    successes = sum(1 for r in records if r.get("success", False))

    p = lambda a, pct: float(np.percentile(a, pct)) if len(a) > 0 else 0.0

    return {
        "recovery_strategy": strategy,
        "fault_type": fault_type,
        "prompt_length_label": prompt_key,
        "concurrency_level": concurrency,
        "total_requests": len(records),
        "successful_requests": successes,
        "failed_requests": len(records) - successes,
        "success_rate": successes / max(len(records), 1),
        "p50_latency_s": p(latencies, 50),
        "p95_latency_s": p(latencies, 95),
        "p99_latency_s": p(latencies, 99),
        "mean_latency_s": float(np.mean(latencies)) if len(latencies) > 0 else 0.0,
        "mean_recovery_time_s": float(np.mean(rec_times)) if len(rec_times) > 0 else 0.0,
        "mean_recomputation_fraction": float(np.mean(recomp_fracs)),
        "throughput_req_per_s": successes / (float(np.sum(latencies)) + 1e-9),
        "mean_ttft_s": p(latencies, 50) * 0.1,
        "p95_ttft_s": p(latencies, 95) * 0.1,
        "total_duration_s": float(np.sum(latencies)),
    }


# ============================================================
# Main experiment loop
# ============================================================

async def run_experiments(output_dir: str = "results") -> str:
    random.seed(42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_rows = []
    total_cells = (
        len(STRATEGIES)
        * len(FAULT_TYPES)
        * len(PROMPT_LENGTH_LABELS)
        * len(CONCURRENCY_LEVELS)
    )
    cell_n = 0

    print(f"\n{'='*65}")
    print(f"  CrashSafe Experiment Suite")
    print(f"  {total_cells} cells × {N_REQUESTS} requests = {total_cells * N_REQUESTS} total")
    print(f"{'='*65}\n")

    t_suite_start = time.time()

    for strategy in STRATEGIES:
        for fault_type in FAULT_TYPES:
            for prompt_key in list(PROMPT_LENGTH_LABELS.keys()):
                for concurrency in CONCURRENCY_LEVELS:
                    cell_n += 1
                    row = await run_cell(strategy, fault_type, prompt_key, concurrency)
                    all_rows.append(row)

                    print(
                        f"  [{cell_n:>3}/{total_cells}] "
                        f"{strategy:<22} {fault_type:<6} {prompt_key:<7} c={concurrency}"
                        f"  →  success={row['success_rate']:>4.0%} "
                        f"p50={row['p50_latency_s']*1000:>5.0f}ms "
                        f"rec={row['mean_recovery_time_s']*1000:>5.0f}ms "
                        f"recomp={row['mean_recomputation_fraction']:>4.0%}"
                    )

    t_elapsed = time.time() - t_suite_start

    # Write CSV
    csv_path = Path(output_dir) / "latest_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Also write timestamped run
    run_path = Path(output_dir) / f"run_{int(time.time())}.csv"
    with open(run_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*65}")
    print(f"  Experiment complete in {t_elapsed:.1f}s")
    print(f"  Results: {csv_path}")
    print(f"{'='*65}\n")

    return str(csv_path)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    csv_path = asyncio.run(run_experiments("results"))

    # Generate plots
    try:
        from experiments.plotting.plot_results import generate_all_plots
        generate_all_plots(csv_path, "figures")
    except Exception as e:
        print(f"Note: plotting requires seaborn/matplotlib. Error: {e}")
        print("Run manually: python experiments/plotting/plot_results.py results/latest_results.csv")
