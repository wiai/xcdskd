:: sphinx-apidoc [options] -o outputdir packagedir [pathnames]
:: http://www.sphinx-doc.org/en/1.5.1/invocation.html#invocation-apidoc

RMDIR "api" /S /Q
sphinx-apidoc -fMe -o api ../src/aloe
DEL "api\modules.rst"
