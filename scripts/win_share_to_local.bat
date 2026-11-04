@echo off
chcp 65001 >nul
REM ================================================================
REM  OpenStat Studio 공유 폴더 실행 전환 (Windows)
REM  사용: call "<프로젝트>\scripts\win_share_to_local.bat" "<프로젝트>\." START_OPENSTAT.bat
REM  - 네트워크 공유 폴더, 네트워크 드라이브, Linux 가상환경이 있는 폴더에서
REM    실행된 경우 프로젝트를 로컬 폴더로 복사(변경분만 갱신)한 뒤
REM    로컬 사본의 같은 스크립트를 실행한다.
REM  - 로컬 폴더 기본값: USERPROFILE\OpenStatStudio
REM    (환경변수 OPENSTAT_LOCAL_DIR로 변경 가능)
REM  - 결과: OPENSTAT_HANDOFF=1이면 호출한 스크립트는 즉시 종료한다.
REM ================================================================
set "OPENSTAT_HANDOFF=0"
set "OPENSTAT_HANDOFF_RC=0"
set "OS_SRC=%~f1"
set "OS_TARGET=%~2"
set "OS_REASON="

if "%OS_SRC:~0,2%"=="\\" set "OS_REASON=네트워크 공유 폴더"
if defined OS_REASON goto :MIRROR
for /f "usebackq delims=" %%T in (`powershell -NoProfile -NonInteractive -Command "(New-Object System.IO.DriveInfo '%OS_SRC:~0,1%').DriveType" 2^>nul`) do if /i "%%T"=="Network" set "OS_REASON=네트워크 드라이브"
if defined OS_REASON goto :MIRROR
if exist "%OS_SRC%\.venv\bin\" if not exist "%OS_SRC%\.venv\Scripts\" set "OS_REASON=Linux용 가상환경이 있는 폴더"
if defined OS_REASON goto :MIRROR
exit /b 0

:MIRROR
if not defined OPENSTAT_LOCAL_DIR set "OPENSTAT_LOCAL_DIR=%USERPROFILE%\OpenStatStudio"
set "OS_DST=%OPENSTAT_LOCAL_DIR%"
if /i "%OS_SRC%"=="%OS_DST%" exit /b 0

echo.
echo ================================================================
echo   [안내] %OS_REASON%에서 실행됨
echo   이 PC의 로컬 폴더로 프로그램을 복사·갱신한 뒤 실행
echo     원본: "%OS_SRC%"
echo     로컬: "%OS_DST%"
echo ================================================================
echo [복사] 변경된 파일만 복사 중...
robocopy "%OS_SRC%" "%OS_DST%" /E /FFT /R:1 /W:1 /NP /NFL /NDL /NJH /NJS /XD "%OS_SRC%\.venv" "%OS_SRC%\.git" "%OS_SRC%\_test" "%OS_SRC%\workspace" "%OS_SRC%\reports" "%OS_SRC%\dist" "%OS_SRC%\build" __pycache__ .pytest_cache *.egg-info /XF .mcp.json *.pyc
if errorlevel 8 goto :COPY_FAILED
REM reports 폴더는 로컬에 없는 파일만 추가하여 이 PC의 결과물을 덮어쓰지 않는다.
if exist "%OS_SRC%\reports\" robocopy "%OS_SRC%\reports" "%OS_DST%\reports" /E /FFT /R:1 /W:1 /NP /NFL /NDL /NJH /NJS /XC /XN /XO
if errorlevel 8 goto :COPY_FAILED
echo [완료] 로컬 사본 준비 완료 - 로컬 사본으로 실행 전환
echo.

call "%OS_DST%\%OS_TARGET%"
set "OPENSTAT_HANDOFF_RC=%errorlevel%"
set "OPENSTAT_HANDOFF=1"
exit /b 0

:COPY_FAILED
echo.
echo [오류] 로컬 폴더 복사 실패
echo   - 공유 폴더 연결 상태와 로컬 폴더 쓰기 권한 확인 필요
echo   - 로컬 폴더 변경: 환경변수 OPENSTAT_LOCAL_DIR 지정
echo.
pause
set "OPENSTAT_HANDOFF_RC=1"
set "OPENSTAT_HANDOFF=1"
exit /b 1
