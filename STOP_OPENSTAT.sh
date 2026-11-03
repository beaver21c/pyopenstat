#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO] Stopping OpenStat Studio processes..."
pkill -f "run_web_server.py" >/dev/null 2>&1 || true

# Check ports 8500 and 8000 via lsof / fuser if available
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8500/tcp >/dev/null 2>&1 || true
    fuser -k 8000/tcp >/dev/null 2>&1 || true
fi

echo "[INFO] OpenStat Studio stopped."
