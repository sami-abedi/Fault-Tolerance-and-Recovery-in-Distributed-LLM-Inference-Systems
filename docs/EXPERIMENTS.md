# Experiment Guide

This document describes the experimental methodology, how to reproduce all experiments, and how to interpret the results.

## Experiment Design

### Variables

**Independent variables** (swept in full factorial design):

| Dimension | Values | Description |
|---|---|---|
| Recovery strategy | fail_stop, retry_from_scratch, token_commit_resume | Core comparison |
| Fault type | none, kill, delay, hang | Fault conditions |
| Concurrency | 1, 4, 8 | Parallel request load |
| Prompt length | short (32 tok), medium (128 tok), long (512 tok) | Request size |

**Dependent variables** (measured per request):

| Metric | Description |
|---|---|
| End-to-end latency | Wall-clock time from request submission to response |
| Time-to-first-token (TTFT) | Time until first token is produced |
| Recovery time | Time spent in fault handling / recovery |
| Tokens recomputed | Tokens regenerated after fault recovery |
| Success rate | Fraction of requests that return a valid response |
| Throughput | Successful requests per second |

### Derived Aggregates

For each experiment cell, we compute:
- P50, P95, P99 latency
- Mean ± standard deviation for all metrics
- Success rate
- Throughput (req/s)

### Statistical Methodology

- 20 requests per cell (configurable via `--requests`)
- Mock backend ensures deterministic token generation across runs
- Fault probability: 30% per request (configurable via `SimulationParams.fault_probability`)
- Random seed for fault injection is per-run

---

## Reproducing Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick Validation (< 60 seconds)

```bash
python experiments/run_experiments.py --mode quick --requests 5 --max-tokens 16
```

Covers: 2 concurrency levels × 3 fault types × 3 strategies × 3 prompt lengths = 54 cells

### Full Experiment Suite (recommended for paper)

```bash
python experiments/run_experiments.py \
    --mode local \
    --requests 20 \
    --max-tokens 64 \
    --output-dir results/ \
    --plot
```

Covers: 3 × 4 × 3 × 3 = 108 cells × 20 requests = 2,160 total requests

Estimated runtime: 5–15 minutes (CPU-only, mock backend)

### Generating Figures Only

```bash
python experiments/plotting/plot_results.py results/latest_results.csv \
    --output-dir figures/
```

### Server-Based Experiment (with real HTTP)

```bash
# Terminal 1: start server
./scripts/start_server.sh

# Terminal 2: run experiments against server
python experiments/run_experiments.py --mode server --requests 20
```

---

## Understanding the Results

### Expected Findings

#### Finding 1: Fail-Stop has lowest overhead but worst reliability

Under any non-zero fault rate, Fail-Stop should show:
- Lowest P50 latency (no retry overhead)
- Lowest success rate (any fault → failure)
- Zero tokens recomputed (not applicable: request just fails)

#### Finding 2: Retry-from-Scratch adds latency but improves reliability

Under kill/hang faults, Retry should show:
- Higher P95/P99 latency (retry contributes to tail)
- Similar success rate to Token-Commit-Resume (with same max_retries)
- High tokens recomputed (full max_new_tokens per failed attempt)
- Recovery time ≈ backoff + full_generation_time

#### Finding 3: Token-Commit-Resume reduces recovery time and recomputation

Under kill/hang faults, Token-Commit-Resume should show:
- Similar P50 latency to Retry (same work on success)
- Lower recovery time than Retry (fewer tokens to regenerate)
- Lower tokens recomputed (≈ max_tokens - committed_tokens)
- Slightly higher baseline latency than Fail-Stop (checkpoint I/O overhead)

### Reading the CSV

The `aggregate_results.csv` has one row per experiment cell:

```
recovery_strategy, fault_type, concurrency_level, prompt_length_label,
total_requests, successful_requests, success_rate,
p50_latency_s, p95_latency_s, p99_latency_s, mean_latency_s,
mean_recovery_time_s, mean_recomputation_fraction,
throughput_req_per_s, ...
```

### Reading the Figures

| Figure | What it shows |
|---|---|
| `fig1_latency_bars.png` | P50 and P95 latency by strategy × fault type |
| `fig2_success_rate_heatmap.png` | Success rate grid across all conditions |
| `fig3_recovery_time.png` | Mean recovery time under fault conditions |
| `fig4_recomputation.png` | Token recomputation fraction |
| `fig5_throughput_concurrency.png` | Throughput scaling with concurrency |
| `fig6_latency_prompt_length.png` | Latency growth with prompt length |
| `fig7_strategy_summary.png` | Multi-metric comparison under kill faults |

---

## Tuning Experiment Parameters

### Mock Backend Speed

The `tokens_per_second` parameter in `SimulationParams` (default: 100) controls how fast tokens are generated. Lower values amplify the difference between recovery strategies.

### Fault Probability

`SimulationParams.fault_probability` (default: 0.3) controls what fraction of requests experience a fault. Set to 1.0 for worst-case analysis.

### Checkpoint Granularity

`checkpoint_every_n` (default: 10) controls how often tokens are flushed to the store. Lower values reduce tokens recomputed but increase I/O overhead.

---

## Simulation vs. Real Server Mode

The default `--mode local` runs entirely in-process without HTTP. This:
- Is faster (no serialization overhead)
- Is more reproducible (no network variability)
- Directly exercises the recovery logic
- Is sufficient for validating the experimental trends

The `--mode server` mode runs against a real FastAPI server via HTTP. This:
- Includes real HTTP/TCP overhead
- Tests the full system including router logic
- Requires a running server (`./scripts/start_server.sh`)
- Can use real model backends (transformers, vllm)

For the paper, local simulation results are appropriate for demonstrating the algorithmic tradeoffs. Server-mode results can validate that HTTP overhead does not fundamentally change the trends.
