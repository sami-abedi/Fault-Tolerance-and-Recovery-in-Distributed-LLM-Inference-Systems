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
┌──────────────────────────────────────────────────────────┐
│                    Client / Experiment                   │
└────────────────────────┬────────────────────────────────┘
                         │ POST /generate
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Router (FastAPI)                       │
└───────────┼────────────────────────────────────────────┘
            │ HTTP /infer
    ┌───────┴──────────────────────────────┐
    │                                       │
    ▼                                       ▼
┌──────────────────┐             ┌──────────────────┐
│  Worker-0 (8100) │             │  Worker-1 (8101) │
└──────────────────┘             └──────────────────┘
            │
            ▼
┌──────────────────────────────────┐
│      Token Checkpoint Store      │
│   (JSONL on disk / in-memory)    │
│   checkpoints/<request_id>.jsonl │
└──────────────────────────────────┘
```

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/sami-abedi/Fault-Tolerance-and-Recovery-in-Distributed-LLM-Inference-Systems.git
cd Fault-Tolerance-and-Recovery-in-Distributed-LLM-Inference-Systems

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Expected Results

| Scenario | Fail-Stop | Retry-from-Scratch | Token-Commit-Resume |
|---|---|---|---|
| No fault | Lowest latency | Similar | Slightly higher (checkpoint I/O) |
| Under kill faults | Worst reliability | Good | Good |
| Recovery time | N/A | ~full_gen_time | ~(max_tokens - k) x token_time |
| Tokens recomputed | N/A | max_new_tokens | max_new_tokens - committed |
| Success rate | Low | High | High |

---

## Paper

LaTeX source: `paper/main.tex`

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Team

**Team 8 — CrashSafe**  
Course: Topics in Distributed Systems (CECS 574)

| Name | GitHub |
|---|---|
| Syed Mohammed Sami Abedi | [@sami-abedi](https://github.com/sami-abedi) |
| Jay Rakesh Patel | — |
