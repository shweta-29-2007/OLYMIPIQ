@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.12 from python.org and enable the py launcher.
    exit /b 1
)

set "OLYMPIQ_ENV=.venv-olympiq312"

if not exist "%OLYMPIQ_ENV%\Scripts\python.exe" (
    echo Ensuring Python 3.12 is installed...
    py install 3.12
    if errorlevel 1 (
        echo Python 3.12 installation failed. Check your internet connection, then run: py install 3.12
        exit /b 1
    )
    py -3.12 -m venv "%OLYMPIQ_ENV%"
    if not exist "%OLYMPIQ_ENV%\Scripts\python.exe" (
        echo Could not create the Python 3.12 environment.
        echo Run this command manually and try again: py install 3.12
        exit /b 1
    )
)

"%OLYMPIQ_ENV%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Review the pip error above, then retry this launcher.
    exit /b 1
)

"%OLYMPIQ_ENV%\Scripts\python.exe" -m streamlit run app.py
endlocal
