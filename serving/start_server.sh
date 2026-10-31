#!/usr/bin/env bash
set -euo pipefail

# SLM Project - Production Server Startup Script
# Usage: ./serving/start_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"
CHECKPOINT="${SLM_CHECKPOINT:-slm-training/best_model.pt}"
API_KEY="${API_KEY:-change-me-to-a-real-key}"

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT" >&2
    echo "Set SLM_CHECKPOINT environment variable to point to a valid .pt file" >&2
    exit 1
fi

# Create log directory
mkdir -p logs

# Export environment variables
export SLM_CHECKPOINT
export API_KEY
export MODEL_DEVICE="${MODEL_DEVICE:-cpu}"
export PYTHONUNBUFFERED=1

echo "Starting SLM API server..."
echo "  Host:      $HOST"
echo "  Port:      $PORT"
echo "  Workers:   $WORKERS"
echo "  Log Level: $LOG_LEVEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Model Device: $MODEL_DEVICE"
echo ""

exec python -m uvicorn slm_api:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --access-log \
    --proxy-headers \
    --forwarded-allow-ips '*'
