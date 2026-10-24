@echo off
setlocal
cd /d "%~dp0"

set "OLYMPIQ_ENV=.venv-olympiq312"
if not exist "%OLYMPIQ_ENV%\Scripts\python.exe" (
    echo Python environment missing. Run run_olympiq.bat once to set it up.
    exit /b 1
)

if not exist "%OLYMPIQ_ENV%\Scripts\streamlit.exe" (
    "%OLYMPIQ_ENV%\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo Node.js and npm are required for the interactive frontend.
    exit /b 1
)
if not exist "node_modules\.bin\vite.cmd" (
    npm.cmd ci
    if errorlevel 1 exit /b 1
)

netstat -ano | findstr /C:":8501 " >nul
if errorlevel 1 (
    start "OLYMPIQ Streamlit dashboard" cmd /k ""%CD%\%OLYMPIQ_ENV%\Scripts\python.exe" -m streamlit run "%CD%\app.py" --server.port 8501"
    timeout /t 3 /nobreak >nul
)

npm.cmd run dev -- --host 127.0.0.1 --open
endlocal
