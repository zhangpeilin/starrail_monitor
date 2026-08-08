@echo off
REM Learn templates from archived frames and show comparison report
REM Double-click to run template learning, then decide in the report window
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Please run start.bat first to setup.
    pause
    exit /b 1
)
echo Starting template learning (full collection + replay compare, about 5-8 min)...
echo After it finishes, a report window will show old vs new templates and accuracy.
"venv\Scripts\python.exe" template_learn.py
if errorlevel 1 (
    echo Learning failed.
    pause
    exit /b 1
)
pause
