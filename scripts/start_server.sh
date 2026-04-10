#!/usr/bin/env bash
# ============================================================
# CrashSafe — Start the distributed inference server
# ============================================================
# Usage:
#   ./scripts/start_server.sh [--config path/to/config.yaml]
#   ./scripts/start_server.sh --backend mock --workers 2
#   ./scripts/start_server.sh --strategy token_commit_resume
#
# The server starts the router on port 8000 and spawns worker
# subprocesses on ports 8100, 8101, ...
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Default values
CONFIG="${CRASHSAFE_CONFIG:-configs/default.yaml}"
HOST="0.0.0.0"
PORT="8000"
LOG_LEVEL="info"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)   CONFIG="$2";    shift 2 ;;
        --host)     HOST="$2";      shift 2 ;;
        --port)     PORT="$2";      shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--config FILE] [--host HOST] [--port PORT] [--log-level LEVEL]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure directories exist
mkdir -p logs results checkpoints figures

echo "============================================================"
echo "  CrashSafe Distributed LLM Inference Server"
echo "============================================================"
echo "  Config:       $CONFIG"
echo "  Host:         $HOST:$PORT"
echo "  Log level:    $LOG_LEVEL"
echo "============================================================"
echo ""

export CRASHSAFE_CONFIG="$CONFIG"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -m src.server.main \
    --config "$CONFIG" \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL"
