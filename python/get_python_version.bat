@echo off
:: Helper script to read WinPython version from config file
:: Usage: call get_python_version.bat
:: Output: Sets WINPYTHON_VERSION environment variable

setlocal enabledelayedexpansion

:: Default version if config file doesn't exist or is invalid
set "DEFAULT_VERSION=WPy64-31370"
set "CONFIG_FILE=%~dp0version_config.txt"
set "WINPYTHON_VERSION=%DEFAULT_VERSION%"

:: Check if config file exists
if exist "%CONFIG_FILE%" (
    echo Reading WinPython version from %CONFIG_FILE%...

    :: Read the last line of config file (which should contain the actual version)
    for /f "usebackq delims=" %%b in ("%CONFIG_FILE%") do set "CONFIG_VERSION=%%b"

    :: Check if we got a version line
    echo !CONFIG_VERSION! | findstr /c:"WINPYTHON_VERSION=" >nul
    if !errorlevel! == 0 (
        :: Extract version from the line
        for /f "tokens=2 delims==" %%v in ("!CONFIG_VERSION!") do (
            set "CONFIG_VERSION=%%v"
            :: Remove any quotes or whitespace
            set "CONFIG_VERSION=!CONFIG_VERSION:"=!"
            for /l %%c in (1,1,31) do if "!CONFIG_VERSION:~-1!"==" " set "CONFIG_VERSION=!CONFIG_VERSION:~0,-1!"

            :: Validate that version looks like a WinPython directory
            echo !CONFIG_VERSION! | findstr /r /c:"WPy64-[0-9]" >nul
            if !errorlevel! == 0 (
                set "WINPYTHON_VERSION=!CONFIG_VERSION!"
                echo Found version: !WINPYTHON_VERSION!
            ) else (
                echo Warning: Invalid version format in config: !CONFIG_VERSION!
                echo Using default version: !WINPYTHON_VERSION!
            )
        )
    ) else (
        echo Warning: No valid WINPYTHON_VERSION found in config file
        echo Using default version: !WINPYTHON_VERSION!
    )
) else (
    echo Warning: Config file not found: %CONFIG_FILE%
    echo Using default version: %WINPYTHON_VERSION%
)

:: Check if the specified version actually exists
set "WINPYTHON_PATH=%~dp0!WINPYTHON_VERSION!"
if exist "!WINPYTHON_PATH!" (
    echo [OK] Version !WINPYTHON_VERSION! is available
) else (
    echo [ERROR] Warning: Version !WINPYTHON_VERSION! not found in python directory
    echo Available versions:
    dir "%~dp0WPy*" /b 2>nul || echo No WinPython directories found
    echo.
    echo To fix this:
    echo 1. Run python\setup_python.bat to install WinPython
    echo 2. Or edit python\version_config.txt to use an existing version
)

:: Export the version for use in calling scripts
endlocal & set "WINPYTHON_VERSION=%WINPYTHON_VERSION%"