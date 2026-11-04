@echo off
chcp 65001 >nul
REM ================================================================
REM  OpenStat Studio 실행 환경 준비 (Windows)
REM  사용: call "<프로젝트>\scripts\win_install_env.bat" "<프로젝트>\."
REM  - .venv가 없으면 Python 3.10 이상으로 생성
REM  - requirements.txt가 마지막 설치 때와 다르면 패키지 설치·갱신
REM  - 결과: errorlevel 0 성공, 1 실패. 변수 VENV_PY 설정
REM ================================================================
set "OS_ROOT=%~f1"
set "VENV_PY=%OS_ROOT%\.venv\Scripts\python.exe"
set "OS_MARK=%OS_ROOT%\.venv\requirements.installed.txt"
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

if exist "%VENV_PY%" goto :CHECK_DEPS

call :FIND_PYTHON
if not defined OS_PY goto :NO_PYTHON
echo.
echo ================================================================
echo   [최초 1회 설정] 가상환경 생성 및 필수 패키지 설치
echo   인터넷 속도에 따라 약 3~10분 소요
echo ================================================================
echo [1/3] 가상환경 생성 중 - %OS_PY%
%OS_PY% -m venv "%OS_ROOT%\.venv"
if not exist "%VENV_PY%" goto :VENV_FAILED

:CHECK_DEPS
fc /b "%OS_ROOT%\requirements.txt" "%OS_MARK%" >nul 2>&1
if not errorlevel 1 exit /b 0
echo [2/3] 필수 패키지 설치·갱신 중 - 수 분 소요
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install -r "%OS_ROOT%\requirements.txt" --quiet
if errorlevel 1 goto :PIP_FAILED
echo [3/3] OpenStat Studio 패키지 등록 중
"%VENV_PY%" -m pip install -e "%OS_ROOT%" --no-deps --quiet
if errorlevel 1 goto :PIP_FAILED
copy /y "%OS_ROOT%\requirements.txt" "%OS_MARK%" >nul
echo [완료] 실행 환경 준비 완료
exit /b 0

:FIND_PYTHON
set "OS_PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :FP_PYTHON
set "OS_PY=py -3"
goto :EOF
:FP_PYTHON
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :EOF
set "OS_PY=python"
goto :EOF

:NO_PYTHON
echo.
echo ================================================================
echo   [오류] Python 3.10 이상 미설치 또는 PATH 미등록
echo ================================================================
echo   - 설치 주소: https://www.python.org/downloads/windows/
echo   - 설치 첫 화면 하단의 "Add python.exe to PATH" 항목 선택 필요
echo   - 설치 후 이 파일을 다시 실행
echo.
set "OS_OPEN=Y"
set /p "OS_OPEN=Python 다운로드 페이지 열기 [Y/N, 기본 Y]: "
if /i not "%OS_OPEN%"=="N" start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:VENV_FAILED
echo.
echo [오류] 가상환경 생성 실패 - Python 설치 상태 확인 필요
pause
exit /b 1

:PIP_FAILED
echo.
echo [오류] 패키지 설치 실패
echo   - 인터넷 연결 확인 후 다시 실행
echo   - 같은 오류 반복 시: Python 3.12 또는 3.13 설치, "%OS_ROOT%\.venv" 폴더 삭제 후 다시 실행
echo.
pause
exit /b 1
