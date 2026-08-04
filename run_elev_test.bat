@echo off
setlocal
cd /d "%~dp0"
echo Running elevated capture test...
"venv\Scripts\python.exe" elev_test.py
echo.
echo Test finished. Please send the elev_result.txt content to the assistant.
pause
