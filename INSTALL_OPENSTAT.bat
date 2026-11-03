@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================================
echo   OpenStat Studio v3.0 — 원클릭 환경 설치 및 세팅 마법사
echo   (Pure-Python Statistical & Visualization Suite, MIT License)
echo ================================================================
echo.

REM 1. 파이썬 설치 여부 감지
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
    echo [오류] 시스템에 Python이 설치되어 있지 않거나 환경변수 PATH에 등록되지 않았습니다.
    echo.
    echo OpenStat Studio를 실행하려면 Python 3.10 이상이 필요합니다.
    echo 공식 웹사이트에서 파이썬을 설치해 주세요:
    echo   👉 https://www.python.org/downloads/
    echo.
    echo ※ 주의: 파이썬 설치 창 맨 아래의 [Add python.exe to PATH] 체크박스를 꼭 선택하세요!
    echo.
    set /p OPEN_WEB="지금 파이썬 다운로드 사이트를 브라우저로 여시겠습니까? (Y/N, 기본값 Y): "
    if "!OPEN_WEB!"=="" set "OPEN_WEB=Y"
    if /i "!OPEN_WEB!"=="Y" start https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [감지 완료] 시스템 파이썬: %PYTHON_EXE%
echo.

REM 2. 설치 모드 선택
echo [설치 프로필 선택]
echo   1. 표준 설치 (Standard): 필수 통계, T-검정, ANOVA, 회귀분석, 차트, 한글/엑셀 보고서 (약 80MB)
echo   2. 전체 설치 (Full 권장): 28종 전체 + 구조방정식(SEM), 패널 계량경제학, 생존분석 (약 250MB)
echo.
set "INSTALL_CHOICE=2"
set /p INSTALL_CHOICE="설치할 모드를 선택하세요 (1 또는 2, 기본값 2 [10초 후 자동 선택]): "

echo.
echo ================================================================
echo   가상환경(.venv) 생성 및 패키지 다운로드를 시작합니다...
echo   인터넷 속도에 따라 약 1~2분이 소요됩니다. 잠시만 기다려 주세요.
echo ================================================================
echo.

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [1/3] 가상환경(.venv) 생성 중...
    "%PYTHON_EXE%" -m venv "%~dp0.venv"
    if not exist "%VENV_PYTHON%" (
        echo [오류] 가상환경 생성 실패. 파이썬 버전을 확인하세요.
        pause
        exit /b 1
    )
)

echo [2/3] pip 최신화 및 필수 과학 패키지 자동 설치 중...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet

if "%INSTALL_CHOICE%"=="1" (
    echo   -> 표준 경량 프로필 설치 중...
    "%VENV_PYTHON%" -m pip install numpy scipy pandas statsmodels pingouin matplotlib seaborn openpyxl python-docx tabulate fastapi uvicorn python-multipart rich --quiet
) else (
    echo   -> 전체 28종 통계 & 계량경제 프로필 설치 중...
    "%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt" --quiet
)

echo [3/3] OpenStat Studio 로컬 라이브러리 등록 중...
"%VENV_PYTHON%" -m pip install -e "%~dp0" --no-deps --quiet

echo.
echo ================================================================
echo   [성공] OpenStat Studio v3.0 설치 및 환경 세팅이 완료되었습니다!
echo ================================================================
echo.
echo   • 일반 실행 : OPENSTAT_실행.bat (더블클릭으로 즉시 실행)
echo   • AI 연동   : SETUP_CLAUDE_CODE.bat (Claude Code MCP 원클릭 연동)
echo   • 사용 설명 : docs\user_guide.html (클릭 투 클릭 실전 매뉴얼)
echo.

set /p RUN_NOW="지금 바로 OpenStat Studio를 실행하시겠습니까? (Y/N, 기본값 Y): "
if "!RUN_NOW!"=="" set "RUN_NOW=Y"
if /i "!RUN_NOW!"=="Y" (
    call "%~dp0START_OPENSTAT.bat"
)

exit /b 0
