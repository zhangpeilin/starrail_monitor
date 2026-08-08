@echo off
REM Sound sample collector: split game recording into sound effect templates
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" goto :check
echo [ERROR] venv not found. Run start.bat first.
pause
exit /b 1
:check
if "%~1"=="" (
    echo Usage: drag a .wav recording of the game audio onto this file.
    echo Or run:  collect_sounds.bat path	oecording.wav
    pause
    exit /b 1
)
"venv\Scripts\python.exe" sound_sample_collector.py "%~1" -i
pause
