:: calling portable WinPython "python" from directory of cmd script
cd /d "%~dp0"

:: Get WinPython version from configuration
call "..\python\get_python_version.bat"

:: Check if version was set correctly
if "%WINPYTHON_VERSION%"=="" (
    echo ERROR: Could not determine WinPython version
    echo Please check python/version_config.txt
    pause
    exit /b 1
)

:: Activate the configured WinPython environment
call "..\python\%WINPYTHON_VERSION%\scripts\env.bat"

:: Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate WinPython environment
    echo Version: %WINPYTHON_VERSION%
    echo Please check that the version exists in python directory
    pause
    exit /b 1
)

:: Install/update the package
uv pip install -e ..\.

:: Start Jupyter notebook
start jupyter notebook
pause