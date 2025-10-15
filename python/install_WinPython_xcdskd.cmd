@echo off
:: Script to download and install WinPython 3.13.7.0slim

set PYTHON_DIR=%~dp0
set WINPYTHON_URL=https://github.com/winpython/winpython/releases/download/16.6.20250620final/Winpython64-3.12.10.1slim.exe
set WINPYTHON_EXE=Winpython64-3.12.10.1slim.exe
set WINPYTHON_VERSION=WPy64-312101

set WINPYTHON_TARGET=%WINPYTHON_VERSION%

echo ========================================
echo WinPython Setup for xcdskd
echo ========================================

if exist "%PYTHON_DIR%%WINPYTHON_TARGET%" (
    echo WinPython Version %PYTHON_DIR%%WINPYTHON_TARGET% already installed. 
    echo If you want to re-install, delete this directory first: %PYTHON_DIR%%WINPYTHON_TARGET%
    goto :end
)

echo.
echo Downloading WinPython...
echo URL: %WINPYTHON_URL%

:: Download using curl (Windows 10 1803+ has curl built-in)
curl -L "%WINPYTHON_URL%" -o "%PYTHON_DIR%%WINPYTHON_EXE%"

:: Alternative download method using bitsadmin (Windows 7+)
if errorlevel 1 (
    echo Retrying with bitsadmin...
    bitsadmin /transfer "WinPythonDownload" "%WINPYTHON_URL%" "%PYTHON_DIR%%WINPYTHON_EXE%"
)

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
echo Installing aloe package and all dependencies...
uv pip install -e .[scientific,dev] --system

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