# CrashSafe: Fault Tolerance and Recovery in Distributed LLM Inference Systems

**Team 8 — Systems Course Term Project**

A research-grade distributed LLM inference system that implements and rigorously evaluates three fault tolerance strategies under realistic failure conditions.

---

## System Overview

Modern LLM inference is computationally expensive and increasingly deployed in distributed multi-node configurations. A single worker crash during token generation wastes all computed tokens and degrades user experience. CrashSafe investigates how different recovery strategies affect the key tradeoff dimensions: latency overhead, request success rate, and wasted computation.

### Three Recovery Strategies

| Strategy | Description | Overhead | Reliability |
|---|---|---|---|
| **Fail-Stop** | No recovery — request fails immediately | Minimal | Worst under faults |
| **Retry-from-Scratch** | Re-submit entire request | High (full regen) | Good with retries |
| **Token-Commit-Resume** | Checkpoint tokens; resume from last commit | Low (partial regen) | Good with checkpoints |

The key contribution is **Token-Commit-Resume**: incrementally persisting generated tokens to durable storage (JSONL on disk) so that after a fault, generation resumes from the last committed token rather than restarting from scratch.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client / Experiment                   │
└────────────────────────┬────────────────────────────────┘
                         │ POST /generate
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Router (FastAPI)                       │
│  ┌────────────────┐  ┌──────────────────────────────┐  │
│  │  InferenceRouter│  │    Recovery Strategy          │  │
│  │  (round-robin, │  │  ┌───────┐ ┌──────┐ ┌──────┐ │  │
│  │   health check) │  │  │Fail   │ │Retry │ │Token │ │  │
│  └────────┬───────┘  │  │Stop   │ │From  │ │Commit│ │  │
│           │          │  │       │ │Scrtch│ │Resume│ │  │
│           │          │  └───────┘ └──────┘ └──────┘ │  │
│           │          └──────────────────────────────┘  │
└───────────┼─────────────────────────────────────────────┘
            │ HTTP /infer
    ┌───────┴──────────────────────────────┐
    │                                       │
    ▼                                       ▼
┌──────────────────┐             ┌──────────────────┐
│  Worker-0 (8100) │             │  Worker-1 (8101) │
│  ┌─────────────┐ │             │  ┌─────────────┐ │
│  │   Backend   │ │             │  │   Backend   │ │
│  │ ┌─────────┐ │ │             │  │ ┌─────────┐ │ │
│  │ │  Mock   │ │ │             │  │ │  Mock   │ │ │
│  │ │  /      │ │ │             │  │ │  /      │ │ │
│  │ │Transf.  │ │ │             │  │ │Transf.  │ │ │
│  │ │  /vLLM  │ │ │             │  │ │  /vLLM  │ │ │
│  │ └─────────┘ │ │             │  │ └─────────┘ │ │
│  └─────────────┘ │             │  └─────────────┘ │
└──────────────────┘             └──────────────────┘
            │
            ▼
┌──────────────────────────────────┐
│      Token Checkpoint Store      │
│   (JSONL on disk / in-memory)    │
│   checkpoints/<request_id>.jsonl │
└──────────────────────────────────┘
```

### Data Flow (Token-Commit-Resume)

```
1. Client → Router: POST /generate {prompt, max_new_tokens}
2. Router → RecoveryStrategy: execute(context, generate_fn)
3. Strategy → TokenStore: initialize(request_id, prompt)
4. Strategy → Worker: POST /infer
5. Worker → Backend: generate(request)
6.     [FAULT OCCURS at token position k]
7. Strategy ← Exception: RuntimeError("crash")
8. Strategy → TokenStore: load(request_id) → checkpoint[0..k-1]
9. Strategy → Worker: POST /infer {resume_from_tokens: [0..k-1]}
10. Worker → Backend: generate(request, resume_from_tokens)
11. Backend: skip tokens [0..k-1], generate [k..N]
12. Strategy → TokenStore: delete(request_id)
13. Router → Client: GenerateResponse {tokens, latency, recovery_info}
```

---

## Repository Structure

```
CrashSafe/
├── src/
│   ├── config.py              # Pydantic configuration (YAML-driven)
│   ├── backends/
│   │   ├── base.py            # Abstract backend interface
│   │   ├── mock_backend.py    # Deterministic CPU backend (no GPU needed)
│   │   ├── transformers_backend.py  # HuggingFace transformers
│   │   ├── vllm_backend.py    # vLLM (GPU, optional)
│   │   └── factory.py         # Backend instantiation
│   ├── recovery/
│   │   ├── base.py            # Abstract recovery strategy
│   │   ├── fail_stop.py       # FAIL_STOP: no recovery
│   │   ├── retry.py           # RETRY_FROM_SCRATCH: full restart
│   │   ├── token_resume.py    # TOKEN_COMMIT_RESUME: checkpoint+resume
│   │   └── factory.py         # Strategy instantiation
│   ├── storage/
│   │   └── token_store.py     # JSONL disk + in-memory stores
│   ├── server/
│   │   ├── main.py            # FastAPI router app + worker launcher
│   │   ├── worker.py          # FastAPI worker app + fault injection
│   │   ├── router.py          # Load-balancing router + health checks
│   │   └── models.py          # Pydantic request/response models
│   └── utils/
│       ├── metrics.py         # Per-request metrics + percentiles
│       ├── logging_utils.py   # Structured JSON logging
│       └── timing.py          # Timing helpers
├── experiments/
│   ├── run_experiments.py     # Full experiment sweep
│   ├── faults/
│   │   └── fault_types.py     # Fault specs + injector client
│   └── plotting/
│       └── plot_results.py    # matplotlib/seaborn figure generation
├── tests/
│   ├── test_backends.py       # Backend unit tests
│   ├── test_recovery.py       # Recovery strategy unit tests
│   └── test_metrics.py        # Metrics collection tests
├── scripts/
│   ├── smoke_test.py          # End-to-end validation (no server needed)
│   ├── start_server.sh        # Server startup script
│   └── run_quick_experiment.sh
├── configs/
│   └── default.yaml           # All system configuration
├── results/                   # Experiment CSV/JSONL outputs
├── figures/                   # Generated plots (PDF + PNG)
├── paper/                     # LaTeX paper source
├── docs/
│   ├── ARCHITECTURE.md        # Detailed design documentation
│   └── EXPERIMENTS.md         # Experiment reproduction guide
└── requirements.txt
```

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Fault-Tolerance-and-Recovery-in-Distributed-LLM-Inference-Systems

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Validation (No Server Required)

```bash
# Smoke test: validates all core components
python scripts/smoke_test.py --verbose

# Quick experiment: runs reduced sweep and generates figures
python experiments/run_experiments.py --mode quick
```

### Run Full Experiment Suite

```bash
# Full experiment (all strategy × fault × concurrency × prompt combinations)
python experiments/run_experiments.py --mode local --requests 20

# Generate plots from existing results
python experiments/plotting/plot_results.py results/latest_results.csv --output-dir figures/
```

### Run the Server (Distributed Mode)

```bash
# Start the router + worker subprocesses
./scripts/start_server.sh

# In another terminal, test the API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Fault tolerance in distributed systems", "max_new_tokens": 50}'

# Check system status
curl http://localhost:8000/admin/status

# Inject a fault (arms the fault on next request)
curl -X POST http://localhost:8000/admin/inject-fault \
  -H "Content-Type: application/json" \
  -d '{"fault_type": "kill"}'
```

### Run Tests

```bash
# All unit tests
pytest tests/ -v

# Specific test module
pytest tests/test_recovery.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Configuration

All system behavior is controlled via `configs/default.yaml`.

Key configuration options:

```yaml
backend:
  type: mock          # mock | transformers | vllm
  
recovery:
  strategy: token_commit_resume   # fail_stop | retry_from_scratch | token_commit_resume
  max_retries: 3
  checkpoint_every_n_tokens: 10

worker:
  num_workers: 2
  worker_base_port: 8100

experiment:
  concurrency_levels: [1, 4, 8]
  fault_types: [none, kill, delay, hang]
  recovery_strategies: [fail_stop, retry_from_scratch, token_commit_resume]
  requests_per_cell: 20
```

---

## API Reference

### POST /generate
Submit an inference request.

```json
{
  "prompt": "The distributed system",
  "max_new_tokens": 128,
  "temperature": 0.7,
  "stream": false
}
```

### GET /health
Liveness probe. Returns `{"status": "ok"}`.

### GET /admin/status
System-wide status: worker health, backend type, recovery strategy.

### POST /admin/inject-fault
Arm a fault on a target worker.
```json
{"fault_type": "kill", "worker_id": "worker-0"}
{"fault_type": "delay", "delay_ms": 500}
{"fault_type": "hang", "hang_duration_s": 5}
```

---

## Expected Results

The system is designed to demonstrate the following:

| Scenario | Fail-Stop | Retry-from-Scratch | Token-Commit-Resume |
|---|---|---|---|
| No fault | Lowest latency | Similar | Slightly higher (checkpoint I/O) |
| Under kill faults | Worst reliability | Good | Good |
| Recovery time | N/A | ~full_gen_time | ~(max_tokens - k) × token_time |
| Tokens recomputed | N/A | max_new_tokens | max_new_tokens - committed |
| Success rate | Low | High | High |

---

## Paper

The research paper is located in `paper/main.tex` (LaTeX source).
Compiled figures from experiments go into `paper/figures/`.

Compile the paper:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Team

Team 8 — CrashSafe  
Course: Topics In Distributed Computing 
Topic :Fault Tolerance and Recovery in Distributed LLM Inference Systems
