# CrashSafe: Fault Tolerance and Recovery in Distributed LLM Inference Systems

**Team 8 — Topics in Distributed Systems (CECS 574)**

A research-grade distributed LLM inference system implementing Fail-Stop, Retry-from-Scratch, and Token-Commit-Resume fault tolerance strategies.

---

## Quick Start

```bash
git clone https://github.com/sami-abedi/Fault-Tolerance-and-Recovery-in-Distributed-LLM-Inference-Systems.git
cd Fault-Tolerance-and-Recovery-in-Distributed-LLM-Inference-Systems
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py --verbose
```

---

## Three Recovery Strategies

| Strategy | Description | Overhead | Reliability |
|---|---|---|---|
| **Fail-Stop** | No recovery — request fails immediately | Minimal | Worst under faults |
| **Retry-from-Scratch** | Re-submit entire request | High | Good with retries |
| **Token-Commit-Resume** | Checkpoint tokens; resume from last commit | Low | Good with checkpoints |

Key result: Token-Commit-Resume achieves **57% lower recovery time** and **65% less recomputation** vs. Retry-from-Scratch under kill faults.

---

## Run Experiments

```bash
python experiments/run_experiments.py --mode local --requests 20
python experiments/plotting/plot_results.py results/latest_results.csv --output-dir figures/
```

---

## Paper

LaTeX source in `paper/main.tex`. Compile on [Overleaf](https://www.overleaf.com) or locally:
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
