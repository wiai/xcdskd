import inspect
from IPython.core.display import display, HTML

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
#from pygments.styles import get_style_by_name

def show_source(obj):
    """
    Display highlighted source code of function in Jupyter notebook.
    
    Use to show source of imported functions etc.
    """
    src=inspect.getsourcelines(obj)
    code=''.join(src[0])
    code_highlighted=highlight(code, PythonLexer(), HtmlFormatter(style='friendly', noclasses=True))
    display(HTML(code_highlighted))
    return