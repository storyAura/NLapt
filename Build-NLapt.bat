@echo off
rem === NLapt quick build (PyInstaller) ===
rem Double-click to build dist\NLapt\NLapt.exe (onedir, windowed, no console).
cd /d "%~dp0"
title NLapt Build

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

rem Ensure build + runtime dependencies are present. Every runtime dependency
rem must be importable at build time: PyInstaller only WARNS about a missing
rem hidden import and would ship an exe whose inference fails with
rem "The 'httpx' package is required for LLM HTTP clients".
%PY% -c "import PyInstaller, PySide6, PIL, httpx, numpy, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing build dependencies: pyinstaller, PySide6, Pillow, httpx, numpy, onnxruntime ...
    %PY% -m pip install pyinstaller PySide6 Pillow httpx numpy onnxruntime
    if errorlevel 1 (
        echo [ERROR] Build dependency install failed. Check your network and retry.
        echo.
        pause
        exit /b 1
    )
)

echo Building NLapt (the first build may take a while) ...
echo.
%PY% -m PyInstaller packaging\nlapt.spec --noconfirm
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo [ERROR] Build failed with code %RC%. See the PyInstaller output above.
    echo.
    pause
    exit /b %RC%
)

echo.
echo Build complete.
echo   Output folder: %cd%\dist\NLapt
echo   Executable:    %cd%\dist\NLapt\NLapt.exe
echo.
if exist "dist\NLapt" start "" "dist\NLapt"
pause
