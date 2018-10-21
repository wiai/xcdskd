::call make_apidoc.bat
python -m sphinx -b latex . ../latex -j4 -E -a
chdir ../latex 
pdflatex xcdskd
pdflatex xcdskd
pdflatex xcdskd
chdir ../doc