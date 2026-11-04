@echo off
chcp 65001 >nul
setlocal
REM ================================================================
REM  OpenStat Studio Claude Code 연동 설정 (Windows)
REM  - 공유 폴더에서 실행 시 로컬 폴더로 복사 후 로컬 사본에서 설정
REM  - 가상환경 준비 후 .mcp.json 생성 및 claude mcp add 등록
REM ================================================================
call "%~dp0scripts\win_share_to_local.bat" "%~dp0." "%~nx0"
if "%OPENSTAT_HANDOFF%"=="1" exit /b %OPENSTAT_HANDOFF_RC%
cd /d "%~dp0"

echo ================================================================
echo   OpenStat Studio v3.0 - Claude Code 연동 설정
echo ================================================================
call "%~dp0scripts\win_install_env.bat" "%~dp0."
if errorlevel 1 exit /b 1
"%VENV_PY%" "%~dp0scripts\setup_claude_code.py"
echo.
pause
exit /b 0
