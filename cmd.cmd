:: Get WinPython version from configuration
call ./python/get_python_version.bat
call "./python/%WINPYTHON_VERSION%/scripts/env.bat"
start cmd.exe