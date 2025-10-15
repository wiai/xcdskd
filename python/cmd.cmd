:: Get WinPython version from configuration
call get_python_version.bat
call ".\%WINPYTHON_VERSION%\scripts\env.bat"
start cmd.exe