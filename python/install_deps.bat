@echo off
:: Install project dependencies using uv

cd /d "%~dp0\.."

:: Get WinPython version from configuration
call python\get_python_version.bat

:: Check if version was set correctly
if "%WINPYTHON_VERSION%"=="" (
    echo ERROR: Could not determine WinPython version
    echo Please check python/version_config.txt
    pause
    exit /b 1
)

:: Activate the configured WinPython environment
call python\%WINPYTHON_VERSION%\scripts\env.bat

:: Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate WinPython environment
    echo Version: %WINPYTHON_VERSION%
    echo Please check that the version exists in python directory
    pause
    exit /b 1
)

echo Using WinPython version: %WINPYTHON_VERSION%
echo.

echo Installing aloe package and dependencies...
uv pip install -e .

echo.
echo Installing optional dependencies...
choice /C YN /M "Install scientific packages (spglib, pymatgen)?"
if errorlevel 1 if not errorlevel 2 uv pip install -e .[scientific]

choice /C YN /M "Install development tools (jupyter, sphinx)?"
if errorlevel 1 if not errorlevel 2 uv pip install -e .[dev]

echo.
echo Installation complete!
pause