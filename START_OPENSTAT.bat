@echo off
chcp 65001 >nul
setlocal
REM ================================================================
REM  OpenStat Studio v3.0 실행기 (Windows)
REM  - 공유 폴더에서 실행 시 로컬 폴더로 복사 후 로컬 사본 실행
REM  - 최초 실행 시 가상환경 생성과 패키지 설치 자동 수행
REM  - 로컬 서버 127.0.0.1:8500 구동 후 앱 창 실행
REM ================================================================
call "%~dp0scripts\win_share_to_local.bat" "%~dp0." "%~nx0"
if "%OPENSTAT_HANDOFF%"=="1" exit /b %OPENSTAT_HANDOFF_RC%
cd /d "%~dp0"

echo ================================================================
echo   OpenStat Studio v3.0 - 오픈소스 통계분석 스튜디오
echo ================================================================

call "%~dp0scripts\win_install_env.bat" "%~dp0."
if errorlevel 1 exit /b 1

set "OS_PORT=8500"
set "OS_URL=http://127.0.0.1:8500"
set "OS_LOG=%~dp0workspace\server.log"
if not exist "%~dp0workspace\" mkdir "%~dp0workspace"

netstat -ano | findstr /r /c:":%OS_PORT% .*LISTENING" >nul
if not errorlevel 1 goto :ALREADY_RUNNING

echo [서버] 로컬 서버 구동 중 - %OS_URL%
if exist "%OS_LOG%" del "%OS_LOG%" >nul 2>&1
set "OPENSTAT_SERVER_LOG=%OS_LOG%"
start "" /b "%VENV_PY%" "%~dp0run_web_server.py" --host 127.0.0.1 --port %OS_PORT%
set "OPENSTAT_SERVER_LOG="

where curl.exe >nul 2>&1
if errorlevel 1 goto :WAIT_FIXED
set /a OS_WAIT=0
:WAIT_LOOP
curl.exe -s -o nul --max-time 2 "%OS_URL%/" >nul 2>&1
if not errorlevel 1 goto :OPEN_APP
findstr /c:"OPENSTAT_SERVER_FATAL" "%OS_LOG%" >nul 2>&1
if not errorlevel 1 goto :SERVER_FAILED
set /a OS_WAIT+=1
if %OS_WAIT% GEQ 120 goto :SERVER_SLOW
timeout /t 1 /nobreak >nul
goto :WAIT_LOOP

:WAIT_FIXED
timeout /t 10 /nobreak >nul
goto :OPEN_APP

:SERVER_SLOW
echo [안내] 서버 응답 지연 - 앱 창을 먼저 실행. 빈 화면이면 잠시 후 새로고침
goto :OPEN_APP

:SERVER_FAILED
echo.
echo [오류] 서버 구동 실패 - 오류 기록:
type "%OS_LOG%"
echo.
pause
exit /b 1

:ALREADY_RUNNING
echo [안내] 서버가 이미 실행 중 - 기존 서버로 연결

:OPEN_APP
echo [화면] 앱 창 실행
set "OS_EDGE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not exist "%OS_EDGE%" set "OS_EDGE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if exist "%OS_EDGE%" goto :OPEN_EDGE
set "OS_CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%OS_CHROME%" set "OS_CHROME=%LocalAppData%\Google\Chrome\Application\chrome.exe"
if exist "%OS_CHROME%" goto :OPEN_CHROME
start "" "%OS_URL%"
goto :LAUNCHED

:OPEN_EDGE
start "" "%OS_EDGE%" --app=%OS_URL%
goto :LAUNCHED

:OPEN_CHROME
start "" "%OS_CHROME%" --app=%OS_URL%
goto :LAUNCHED

:LAUNCHED
echo.
echo ================================================================
echo   OpenStat Studio 실행 완료 - %OS_URL%
echo   - 이 창에서 아무 키나 누르거나 창을 닫으면 서버 종료
echo   - 설치 위치: "%~dp0"
echo   - 서버 기록: "%OS_LOG%"
echo ================================================================
echo.
pause
call "%~dp0STOP_OPENSTAT.bat"
exit /b 0
