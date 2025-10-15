:: calling portable WinPython "python" from directory of cmd script
cd /d "%~dp0"
call "..\python\WPy64-312101\scripts\env.bat"
python -m pip install -e ..\.
start jupyter notebook
pause