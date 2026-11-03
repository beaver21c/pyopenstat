#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "  OpenStat Studio v3.0 (MIT Open-Source Python Suite)"
echo "  - Pure-Python Statistical Engine & Customizable Plot Studio"
echo "================================================================"

# 1. Detect system Python 3
PYTHON_SYS=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_SYS="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_SYS="python"
fi

if [ -z "$PYTHON_SYS" ]; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10+ (macOS: brew install python / Linux: sudo apt install python3 python3-venv)"
    exit 1
fi

# 2. Check virtual environment & self-bootstrap if missing
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo ""
    echo "================================================================"
    echo "  [First-Time Setup] Bootstrapping virtual environment (.venv)..."
    echo "  Installing required statistical & plotting packages (~1-2 min)."
    echo "================================================================"
    echo "[1/3] Creating virtual environment (.venv)..."
    "$PYTHON_SYS" -m venv "$SCRIPT_DIR/.venv"
    echo "[2/3] Installing dependencies..."
    "$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip --quiet
    "$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet
    "$SCRIPT_DIR/.venv/bin/pip" install -e "$SCRIPT_DIR" --no-deps --quiet
    echo "[3/3] Setup complete!"
    echo ""
fi

# 3. Clean any stale port 8500 listener before starting
"$SCRIPT_DIR/STOP_OPENSTAT.sh" >/dev/null 2>&1 || true

# 4. Open browser after short delay
(
    sleep 2
    TARGET="http://localhost:8500"
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$TARGET" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
        open "$TARGET" >/dev/null 2>&1 || true
    fi
) &

echo "Starting OpenStat Studio on http://localhost:8500 ..."
echo "Press Ctrl+C to stop the server cleanly."
exec "$VENV_PYTHON" "$SCRIPT_DIR/run_web_server.py" "$@"
