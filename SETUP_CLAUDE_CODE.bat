@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================================
echo   OpenStat Studio v3.0 — Claude Code & AI Agent 1-Click Setup
echo ================================================================

REM 1. 파이썬 설치 여부 확인
set "PYTHON_EXE="
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('where python') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
    )
)
if not defined PYTHON_EXE (
    where py >nul 2>&1
    if %ERRORLEVEL% EQU 0 set "PYTHON_EXE=py"
)

if not defined PYTHON_EXE (
    echo.
    echo [오류] 시스템에 Python이 설치되어 있지 않거나 PATH에 등록되지 않았습니다.
    echo Python 3.10 이상을 설치하신 후 다시 실행해 주세요.
    echo   👉 https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM 2. 설정 스크립트 실행
"%PYTHON_EXE%" "%~dp0scripts\setup_claude_code.py"

echo.
pause
exit /b 0
