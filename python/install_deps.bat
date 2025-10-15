@echo off
:: Install project dependencies using uv

cd /d "%~dp0\.."
call python\WPy64-31370\scripts\env.bat

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