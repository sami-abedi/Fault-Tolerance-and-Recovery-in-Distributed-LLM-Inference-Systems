#!/usr/bin/env bash
# ============================================================
# CrashSafe — Run a quick experiment (no server required)
# ============================================================
# Runs a reduced sweep for rapid validation and demonstration.
# Results are saved to ./results/ and figures to ./figures/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p results figures logs

echo "============================================================"
echo "  CrashSafe Quick Experiment"
echo "============================================================"
echo ""

python experiments/run_experiments.py \
    --mode quick \
    --requests 10 \
    --max-tokens 32 \
    --output-dir results \
    --plot

echo ""
echo "Experiment complete. Results in ./results/, figures in ./figures/"
