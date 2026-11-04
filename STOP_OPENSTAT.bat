@echo off
chcp 65001 >nul
setlocal
REM OpenStat Studio 서버 종료: 8500 포트를 점유한 python 프로세스만 종료
set "OS_PORT=8500"
set "OS_FOUND=0"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%OS_PORT% .*LISTENING"') do call :STOP_PID %%P
if "%OS_FOUND%"=="0" echo [안내] 실행 중인 OpenStat Studio 서버 없음
if "%OS_FOUND%"=="1" echo [완료] OpenStat Studio 서버 종료
exit /b 0

:STOP_PID
tasklist /fi "PID eq %~1" /nh | findstr /i "python" >nul
if errorlevel 1 goto :EOF
taskkill /f /t /pid %~1 >nul 2>&1
set "OS_FOUND=1"
goto :EOF
