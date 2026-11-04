#!/usr/bin/env python3
"""
OpenStat Studio v3.0 — 1-Click Claude Code & AI Agent Setup Tool
- Automatically configures local MCP (Model Context Protocol) for Claude Code, Antigravity, Cursor, and Codex
- Detects OS, venv paths, and generates clean .mcp.json
- Registers with `claude mcp add` if Claude Code CLI is present
"""

import os
import sys
import json
import shutil
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECONSTRUCTED_DIR = os.path.join(PROJECT_ROOT, "reconstructed")

is_win = sys.platform.startswith("win")
venv_bin_dir = "Scripts" if is_win else "bin"
venv_exe_name = "openstat-mcp.exe" if is_win else "openstat-mcp"
venv_python_name = "python.exe" if is_win else "python"

venv_dir = os.path.join(PROJECT_ROOT, ".venv")
venv_mcp = os.path.join(venv_dir, venv_bin_dir, venv_exe_name)
venv_python = os.path.join(venv_dir, venv_bin_dir, venv_python_name)


def main():
    print("=" * 68)
    print("  OpenStat Studio v3.0 — AI Agent & Claude Code Setup")
    print("  (Model Context Protocol: openstat-mcp)")
    print("=" * 68)

    # 1. Check or bootstrap .venv
    if not os.path.isfile(venv_python):
        print(f"[INFO] Virtualenv not found at {venv_dir}. Creating now...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        pip_path = os.path.join(venv_dir, venv_bin_dir, "pip.exe" if is_win else "pip")
        req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
        print("[INFO] Installing OpenStat dependencies into virtualenv...")
        subprocess.run([pip_path, "install", "--upgrade", "pip", "--quiet"], check=False)
        subprocess.run([pip_path, "install", "-r", req_file, "--quiet"], check=False)
        subprocess.run([pip_path, "install", "-e", PROJECT_ROOT, "--no-deps", "--quiet"], check=False)

    # Determine execution command
    if os.path.isfile(venv_mcp):
        cmd = venv_mcp
        args = []
    else:
        # Fallback to python -m openstat.mcp_server
        cmd = venv_python
        args = ["-m", "openstat.mcp_server"]

    env = {"PYTHONPATH": RECONSTRUCTED_DIR}
    if is_win:
        # Korean Windows defaults to cp949; force UTF-8 so Korean files and labels load correctly.
        env["PYTHONUTF8"] = "1"

    mcp_config = {
        "mcpServers": {
            "openstat": {
                "command": cmd,
                "args": args,
                "env": env,
            }
        }
    }

    # 2. Write local .mcp.json in PROJECT_ROOT
    mcp_json_path = os.path.join(PROJECT_ROOT, ".mcp.json")
    with open(mcp_json_path, "w", encoding="utf-8") as f:
        json.dump(mcp_config, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Updated project MCP configuration: {mcp_json_path}")

    # 3. Check claude CLI in PATH
    claude_cmd = shutil.which("claude")
    if claude_cmd:
        print(f"[OK] Claude Code CLI detected at: {claude_cmd}")
        print("[INFO] Registering 'openstat' toolset into Claude Code...")
        try:
            # claude mcp add openstat <command> ...
            register_cmd = [claude_cmd, "mcp", "add", "openstat", cmd] + args
            subprocess.run(register_cmd, check=False)
            print("[OK] Claude Code registration command executed.")
        except Exception as e:
            print(f"[NOTICE] Could not run automatic 'claude mcp add': {e}")
    else:
        print("[NOTICE] 'claude' CLI not in PATH (Claude Code may be used via desktop app or project folder).")
        print("         The local .mcp.json in this directory will be automatically read by Claude Code!")

    print("\n" + "=" * 68)
    print("  [완료] OpenStat Studio와 Claude Code의 연동 준비가 완료되었습니다!")
    print("=" * 68)
    print("\n▶ [사용 방법]")
    print(f"  1. 이 프로젝트 폴더({PROJECT_ROOT})에서 터미널을 열고 'claude'를 실행합니다.")
    print("  2. Claude Code에게 평소 대화하듯 통계 분석 및 시각화를 지시하세요:")
    print("     - \"ToothGrowth.csv 데이터 불러와서 dose별 t-검정하고 APA 표 출력해줘\"")
    print("     - \"Grunfeld_Panel.csv 패널 데이터 고정효과 분석하고 결과 그래프 그려줘\"")
    print("     - \"분석 결과와 그래프를 한글(HWPX)과 엑셀(XLSX) 보고서 파일로 내보내줘\"")
    print("\n▶ [연동 상태 확인]")
    print("  터미널에서 'claude mcp list' 를 입력하면 'openstat' 도구가 표시됩니다.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
