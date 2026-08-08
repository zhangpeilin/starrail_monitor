@echo off
REM WASAPI process-loopback capture self-test.
REM Run this while the game is open and playing sound, then listen to capture_test.wav.
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run start.bat first.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" sound_trigger.py --capture-test StarRail.exe 5
pause
