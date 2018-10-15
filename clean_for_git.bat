REM @echo off

RMDIR /S /Q ".doctrees"
RMDIR /S /Q "html"

del /S /F *.pyc

set dir=%~dp0
FOR /D /R %dir% %%X IN (*__pycache__) DO RMDIR /S /Q "%%X"
FOR /D /R %dir% %%X IN (*.ipynb_checkpoints) DO RMDIR /S /Q "%%X"
pause
exit