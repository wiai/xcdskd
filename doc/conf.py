#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Use sphinx-quickstart to create your own conf.py file!
# After that, you have to edit a few things.  See below.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
path_doc=os.path.abspath('.')
print(path_doc)
sys.path.insert(0,path_doc)

# parent dir of xcdskd package (with aloe/__init__.py)
path_aloe=os.path.join(os.path.abspath(os.pardir))+"/src"
print('Parent dir to aloe package: ', path_aloe)
sys.path.insert(0,path_aloe)



# Select nbsphinx and, if needed, add a math extension (mathjax or pngmath):
extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
]


#extensions += [
#    'matplotlib.sphinxext.only_directives',
#    'matplotlib.sphinxext.plot_directive']


autosummary_generate = False
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

import sphinx_rtd_theme

#html_theme_path = sphinx_rtd_theme.html_theme_path()
html_theme = 'sphinx_rtd_theme'

# Register the theme as an extension to generate a sitemap.xml
extensions.append("sphinx_rtd_theme")

# Exclude build directory and Jupyter backup files:
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Default language for syntax highlighting in reST and Markdown cells
highlight_language = 'none'

# Don't add .txt suffix to source files (available for Sphinx >= 1.5):
html_sourcelink_suffix = ''

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
#nbsphinx_execute = 'never'

# Use this kernel instead of the one stored in the notebook metadata:
#nbsphinx_kernel_name = 'python3'

# List of arguments to be passed to the kernel that executes the notebooks:
#nbsphinx_execute_arguments = ['--InlineBackend.figure_formats={"png", "pdf"}']

# If True, the build process is continued even if an exception occurs:
#nbsphinx_allow_errors = True

# Controls when a cell will time out (defaults to 30; use -1 for no timeout):
#nbsphinx_timeout = 60

# Default Pygments lexer for syntax highlighting in code cells:
#nbsphinx_codecell_lexer = 'ipython3'

# Width of input/output prompts used in CSS:
#nbsphinx_prompt_width = '8ex'

# If window is narrower than this, input/output prompts are on separate lines:
#nbsphinx_responsive_width = '700px'

# -- The settings below this line are not specific to nbsphinx ------------

master_doc = 'index'

project = 'xcdskd'
author = 'Aimo Winkelmann'
copyright = '2018, ' + author

linkcheck_ignore = [r'http://localhost:\d+/']

# -- Get version information from Git -------------------------------------

try:
    from subprocess import check_output
    release = check_output(['git', 'describe', '--tags', '--always'])
    release = release.decode().strip()
except Exception:
    release = '<unknown>'

# -- Options for HTML output ----------------------------------------------

html_title = project + ' version ' + release

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'preamble': r"""
\usepackage[sc,osf]{mathpazo}
\linespread{1.05}  % see http://www.tug.dk/FontCatalogue/urwpalladio/
\renewcommand{\sfdefault}{pplj}  % Palatino instead of sans serif
\IfFileExists{zlmtt.sty}{
    \usepackage[light,scaled=1.05]{zlmtt}  % light typewriter font from lmodern
}{
    \renewcommand{\ttdefault}{lmtt}  % typewriter font from lmodern
}
""",
}

latex_documents = [
    (master_doc, 'xcdskd.tex', project, author, 'howto'),
]

latex_show_urls = 'footnote'



