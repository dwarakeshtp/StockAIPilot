#!/usr/bin/env bash
# Start the FastAPI backend and the Streamlit dashboard together.
#   scripts/dev.sh                 # API on :8000, dashboard on :8501
#   API_PORT=9000 scripts/dev.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
export STOCKPILOT_API_URL="${STOCKPILOT_API_URL:-http://localhost:${API_PORT}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

"$PYTHON" -m uvicorn src.api.main:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM

echo "Waiting for the API (the first start loads the database) ..."
for _ in $(seq 1 180); do
  if curl -fs "${STOCKPILOT_API_URL}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API failed to start" >&2
    exit 1
  fi
  sleep 1
done
echo "API ready: ${STOCKPILOT_API_URL}/docs"

"$PYTHON" -m streamlit run dashboard/app.py --server.port "$UI_PORT"
