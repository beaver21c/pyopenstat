#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

"$PYTHON_SYS" "$SCRIPT_DIR/scripts/setup_claude_code.py" "$@"
