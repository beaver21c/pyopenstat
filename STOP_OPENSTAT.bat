@echo off
chcp 65001 >nul
echo [INFO] Stopping OpenStat Studio server on port 8500 and 8000 ...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8500" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo [INFO] OpenStat Studio server stopped cleanly.
