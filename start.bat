@echo off
REM Self-elevate to match the game's admin privileges (required for window capture)
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
echo Running as administrator.
setlocal
cd /d "%~dp0"
title StarRail Turn/Action Monitor

REM === Step 1: Python venv & dependencies ===
if exist "venv\Scripts\python.exe" goto :tess
echo [First run] Creating Python venv...
python -m venv venv
if errorlevel 1 goto :venv_fail
echo [First run] Installing dependencies (Aliyun mirror)...
"venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if errorlevel 1 goto :venv_fail
echo Dependencies ready.

:tess
REM === Step 2: Tesseract OCR engine (digits only) ===
if exist "%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe" goto :run
echo [First run] Downloading Tesseract OCR engine (~50MB)...
curl -s -x http://127.0.0.1:7890 -L -o "%TEMP%\tesseract-setup.exe" "https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
if errorlevel 1 curl -s -L -o "%TEMP%\tesseract-setup.exe" "https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
"%TEMP%\tesseract-setup.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="%LOCALAPPDATA%\Programs\Tesseract-OCR"
if exist "%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe" goto :run
echo Silent install failed, trying 7-Zip portable extract...
if exist "C:\Program Files\7-Zip\7z.exe" "C:\Program Files\7-Zip\7z.exe" x -y -o"%LOCALAPPDATA%\Programs\Tesseract-OCR" "%TEMP%\tesseract-setup.exe" >/dev/null
if exist "%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe" goto :run
echo [WARN] Tesseract not available, falling back to Windows OCR (single digits unreliable)

:run
REM === Step 3: launch monitor ===
"venv\Scripts\python.exe" starrail_monitor.py
if not errorlevel 1 goto :eof
echo Program exited with error code %errorlevel%
pause
goto :eof

:venv_fail
echo Failed to create venv or install dependencies. Make sure Python 3.10+ is installed and on PATH.
pause
goto :eof
