@echo off
:: Script to download and install WinPython 3.13.7.0slim

set PYTHON_DIR=%~dp0
set WINPYTHON_URL=https://github.com/winpython/winpython/releases/download/17.2.20250920final/WinPython64-3.13.7.0slim.exe
set WINPYTHON_EXE=WinPython64-3.13.7.0slim.exe
set WINPYTHON_TARGET=WPy64-31370

echo ========================================
echo WinPython Setup for xcdskd
echo ========================================

if exist "%PYTHON_DIR%%WINPYTHON_TARGET%" (
    echo WinPython already installed at: %PYTHON_DIR%%WINPYTHON_TARGET%
    choice /C YN /M "Do you want to reinstall?"
    if errorlevel 2 goto :end
    rmdir /S /Q "%PYTHON_DIR%%WINPYTHON_TARGET%"
)

echo.
echo Downloading WinPython...
echo URL: %WINPYTHON_URL%

:: Download using PowerShell
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityPointManager]::Tls12; Invoke-WebRequest -Uri '%WINPYTHON_URL%' -OutFile '%PYTHON_DIR%%WINPYTHON_EXE%'}"

if not exist "%PYTHON_DIR%%WINPYTHON_EXE%" (
    echo ERROR: Download failed!
    pause
    exit /b 1
)

echo.
echo Extracting WinPython...
"%PYTHON_DIR%%WINPYTHON_EXE%" -y -o"%PYTHON_DIR%"

:: Clean up installer
del "%PYTHON_DIR%%WINPYTHON_EXE%"

if not exist "%PYTHON_DIR%%WINPYTHON_TARGET%" (
    echo ERROR: Extraction failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo WinPython installed successfully!
echo Location: %PYTHON_DIR%%WINPYTHON_TARGET%
echo ========================================
echo.
echo Installing project with uv...

:: Install uv in WinPython
call "%PYTHON_DIR%%WINPYTHON_TARGET%\scripts\env.bat"
pip install uv

:: Navigate to project root
cd /d "%~dp0\.."

:: Install project with dependencies
uv pip install -e .

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To use this Python environment, run:
echo   python\run.cmd
echo.

:end
pause