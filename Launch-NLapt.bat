@echo off
rem === NLapt quick launch ===
rem Double-click to start the GUI. The window stays open if an error occurs.
cd /d "%~dp0"
title NLapt Launcher

rem Pick a Python interpreter (prefer the py launcher, else python).
set "PY=python"
where py >nul 2>nul && set "PY=py -3"

%PY% --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ and add it to PATH.
    echo         Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

rem Check required runtime dependencies (PySide6 / Pillow / httpx); auto-install if missing.
rem httpx drives every LLM, translate and local llama-server request - without it
rem inference fails with "The 'httpx' package is required for LLM HTTP clients".
%PY% -c "import PySide6, PIL, httpx" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing runtime dependencies: PySide6, Pillow, httpx ...
    %PY% -m pip install PySide6 Pillow httpx
    if errorlevel 1 (
        echo [ERROR] Dependency install failed. Check your network and retry.
        echo.
        pause
        exit /b 1
    )
)

rem Optional local-inference dependencies (Florence-2 ONNX engine): best effort,
rem a failure here must not block the launch - the app reports it in the UI.
%PY% -c "import numpy, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing optional local-inference dependencies: numpy, onnxruntime ...
    %PY% -m pip install numpy onnxruntime
    if errorlevel 1 (
        echo [WARN] Optional dependency install failed. Florence-2 local tagging will be unavailable.
        echo.
    )
)

echo Starting NLapt ...
%PY% -m nlapt_gui
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo [ERROR] NLapt exited with code %RC%.
    echo         Logs are under your user folder: NLapt\logs
    echo.
    pause
    exit /b %RC%
)
