#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "  OpenStat Studio v3.0 — Installation & Environment Setup"
echo "  (Pure-Python Statistical & Visualization Suite, MIT License)"
echo "================================================================"

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

echo "[Found] System Python: $PYTHON_SYS"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[1/3] Creating virtual environment (.venv)..."
    "$PYTHON_SYS" -m venv "$SCRIPT_DIR/.venv"
fi

echo "[2/3] Installing dependencies..."
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip --quiet
"$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet
"$SCRIPT_DIR/.venv/bin/pip" install -e "$SCRIPT_DIR" --no-deps --quiet

echo "[3/3] Setup complete!"
echo "================================================================"
echo "  OpenStat Studio v3.0 is ready to use!"
echo "  - Desktop UI: ./START_OPENSTAT.sh"
echo "  - AI Setup  : ./SETUP_CLAUDE_CODE.sh"
echo "================================================================"
