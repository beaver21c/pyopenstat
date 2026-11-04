@echo off
chcp 65001 >nul
setlocal
REM ================================================================
REM  OpenStat Studio v3.0 설치 (Windows)
REM  - 가상환경 생성 및 requirements.txt 전체 패키지 설치·갱신
REM  - 공유 폴더에서 실행 시 로컬 폴더로 복사 후 로컬 사본에서 설치
REM ================================================================
call "%~dp0scripts\win_share_to_local.bat" "%~dp0." "%~nx0"
if "%OPENSTAT_HANDOFF%"=="1" exit /b %OPENSTAT_HANDOFF_RC%
cd /d "%~dp0"

echo ================================================================
echo   OpenStat Studio v3.0 설치
echo ================================================================
REM 설치 기록을 지워 패키지 설치·갱신을 다시 수행
if exist "%~dp0.venv\requirements.installed.txt" del "%~dp0.venv\requirements.installed.txt" >nul 2>&1
call "%~dp0scripts\win_install_env.bat" "%~dp0."
if errorlevel 1 exit /b 1

echo.
echo ================================================================
echo   [완료] OpenStat Studio 설치 완료
echo   - 실행: OPENSTAT_실행.bat
echo   - Claude Code 연동: SETUP_CLAUDE_CODE.bat
echo   - 사용 설명서: docs\user_guide.html
echo   - 설치 위치: "%~dp0"
echo ================================================================
echo.
set "OS_RUN=Y"
set /p "OS_RUN=지금 실행 [Y/N, 기본 Y]: "
if /i "%OS_RUN%"=="N" exit /b 0
call "%~dp0START_OPENSTAT.bat"
exit /b %errorlevel%
