RMDIR "api" /S /Q
RMDIR "../html" /S /Q
RMDIR "../.doctrees" /S /Q

call make_apidoc.bat

python -m sphinx -b html . ../html -j4 -E -a -d ../.doctrees
