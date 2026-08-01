@echo off
title Audio Cutter Pro + Video ^& Image Studio
color 0A
echo ======================================================================
echo             🎵 Audio Cutter Pro + Video ^& Image Studio 🚀
echo ======================================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Setting up...
    python -m venv venv
    .\venv\Scripts\pip.exe install -r requirements.txt
    echo [!] Pre-downloading AI models for 100%% offline usage...
    .\venv\Scripts\python.exe download_models.py
)

echo [✓] Starting local server at http://127.0.0.1:5000
echo [✓] Opening web browser...
echo.

start "" cmd /c "timeout /t 2 >nul && start http://127.0.0.1:5000"

.\venv\Scripts\python.exe app.py

pause
