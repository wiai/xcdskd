CONVERSION OF TSL "ANG-FILE + IMAGE-FOLDER" EBSD DATA TO H5OINA

Usage:
click run.cmd

Information:
As the only extra, you need to know the SEM beam voltage of the EBSD map, as this is not available in the ang file.

This file assumes that the TSL measurement data for each single EBSD map is in a separate folder,
with an ang file and a subfolder with images, e.g. a structure something like this:
data1/
data1/data1.ang
data1/data1/data1_x20y30.tif, + all other patterns

The files should be found automatically by the program.

