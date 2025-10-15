xcdskd: Notebooks on Kikuchi Diffraction Methods
================================================

.. image:: https://zenodo.org/badge/153128196.svg
   :target: https://zenodo.org/badge/latestdoi/153128196
.. image:: https://readthedocs.org/projects/xcdskd/badge/?version=latest
   :target: https://xcdskd.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

|  

The **xcdskd** project provides a collection of Jupyter notebooks which discuss and explain a number of 
data analysis approaches which are applicable in the context of Kikuchi Diffraction Methods, 
predominantly Electron Backscatter Diffraction (EBSD), 
Electron Channeling Patterns (ECP), and Electron Channeling Contrast Imaging (ECCI) in the scanning electron microscope (SEM). 

The notebooks document the application of the **aloe** package, which bundles the central functionalities
for Kikuchi pattern analysis. Because the notebooks contain working Python code, users can directly adapt them to 
their own needs and run the modified notebooks on the provided example data, as well as on their own data sets.

The focus of this project is more on the clear explanation of key concepts and work flows, 
and less on how to reach the ultimate optimization of numerical efficiency.
The project also aims to provide open, documented reference algorithms and example data as a benchmark
and basis for further developments.


Installation
------------

Using the portable WinPython distribution:

1. Download and install WinPython run (doubleclick):

    ./python/install_WinPython_xcdskd.cmd

2. This installs Python 3.12 and all dependencies automatically.

3. To use the environment::

    python\run.cmd

Installing dependencies of aloe on any Python system:
on xcdskd root dir: 

in venv:
uv pip install -e .

or system wide
uv pip install -e . --system

Build the html documentation::

    cd doc
    rebuild_html.cmd
    
