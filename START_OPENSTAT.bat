@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================================
echo   OpenStat Studio v3.0 (MIT Open-Source Python Suite)
echo   - 28종 순수 파이썬 통계분석 및 15종 학술 차트 스튜디오
echo ================================================================

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
    echo.
    echo ================================================================
    echo   [오류] 시스템에 Python이 설치되어 있지 않거나 환경변수 PATH에 등록되지 않았습니다.
    echo ================================================================
    echo.
    echo OpenStat Studio를 실행하려면 Python 3.10 이상이 필요합니다.
    echo 공식 웹사이트에서 파이썬을 설치해 주세요:
    echo   👉 https://www.python.org/downloads/
    echo.
    echo ※ 주의: 파이썬 설치 창 맨 아래의 [Add python.exe to PATH] 체크박스를 꼭 선택하세요!
    echo.
    set /p OPEN_WEB="지금 파이썬 다운로드 페이지를 브라우저로 여시겠습니까? (Y/N, 기본값 Y): "
    if "!OPEN_WEB!"=="" set "OPEN_WEB=Y"
    if /i "!OPEN_WEB!"=="Y" start https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM 2. 가상환경(.venv) 확인 및 최초 1회 자동 부트스트랩
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo.
    echo ================================================================
    echo   [최초 1회 실행 환경 자동 설정]
    echo   가상환경(.venv) 생성 및 필수 라이브러리를 자동으로 설치합니다.
    echo   인터넷 환경에 따라 약 1~2분이 소요됩니다. 잠시만 기다려 주세요...
    echo ================================================================
    echo.
    echo [1/3] 가상환경(.venv) 생성 중...
    "%PYTHON_EXE%" -m venv "%~dp0.venv"
    if not exist "%VENV_PYTHON%" (
        echo [오류] 가상환경 생성에 실패했습니다. 파이썬 설치 버전을 확인해 주세요.
        pause
        exit /b 1
    )
    echo [2/3] pip 최적화 및 필수 패키지 자동 설치 중...
    "%VENV_PYTHON%" -m pip install --upgrade pip --quiet
    "%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt" --quiet
    "%VENV_PYTHON%" -m pip install -e "%~dp0" --no-deps --quiet
    echo [3/3] 통계 및 시각화 패키지 구성 완료!
    echo.
)

REM 3. 로컬 온디맨드 서버 구동
netstat -aon | findstr ":8500" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [안내] 이미 OpenStat Studio 서버가 실행 중입니다.
) else (
    echo [1/2] 로컬 서버 구동 중 (http://127.0.0.1:8500)...
    start "OpenStat Studio Server" /MIN cmd /c "set PORT=8500 && "%VENV_PYTHON%" "%~dp0run_web_server.py""
    timeout /t 2 /nobreak >nul
)

REM 4. 앱 창으로 브라우저 실행
set "TARGET_URL=http://127.0.0.1:8500"
echo [2/2] OpenStat Studio 데스크톱 화면을 띄웁니다...
where msedge >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start "" msedge --app=%TARGET_URL%
    goto :LAUNCHED
)
if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" (
    start "" "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" --app=%TARGET_URL%
    goto :LAUNCHED
)
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
    start "" "%ProgramFiles%\Google\Chrome\Application\chrome.exe" --app=%TARGET_URL%
    goto :LAUNCHED
)
start "" %TARGET_URL%

:LAUNCHED
echo.
echo ================================================================
echo   OpenStat Studio v3.0 이 성공적으로 실행되었습니다!
echo   - 로컬 접속 주소: %TARGET_URL%
echo.
echo   [안내] 이 검은색 콘솔 창을 닫거나 아래에서 아무 키나 누르시면
echo   백그라운드 서버가 안전하게 자동 종료됩니다.
echo ================================================================
echo.
pause
call "%~dp0STOP_OPENSTAT.bat"
exit /b 0
