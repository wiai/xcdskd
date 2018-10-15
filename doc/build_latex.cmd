::call make_apidoc.bat
python -m sphinx -b latex . ../latex -j4 -E -a
chdir ../latex 
pdflatex aloha
pdflatex aloha
pdflatex aloha
chdir ../doc