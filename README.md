xcdskd: Notebooks on Kikuchi Diffraction Methods
================================================

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


Standalone Installation with a Portable WinPython
-------------------------------------------------

1. On the current GitHub page, goto the button "<> Code" and select "Download ZIP", this will download "xcdskd-main.zip"

2. Unpack "xcdskd-main.zip" to a working directory on your machine

3. Install WinPython, in the subdirectory "python" doubleclick:

    ./python/install_WinPython_xcdskd.cmd

   This installs Python 3.12 and all dependencies for xcdskd automatically.

4. The applications in "apps" should now recognize the WinPython installation and should be runnable by clicking the respective "run.cmd"

Start a command line by clicking "cmd.cmd", this will use the WinPython installation that was installed above.
You can start arbitrary Python code with the WinPython installation by calling "python\run.cmd some_code.py".

Manual install of dependencies of aloe using uv:

in a uv virtual environment:  uv pip install -e .
system wide                :  uv pip install -e . --system

