:: calling example of portable WinPython "python" from directory of cmd script
cd /d "%~dp0"
call ".\WPy64-31370\scripts\env.bat"
python %*
pause
