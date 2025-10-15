import sys
import h5py

filename_h5oina  = b"t:\\EBSD_Data\\STDEV\\Projects\\AGH\\QCMeteorite\\Ubersicht_10 mit approximant.h5oina"

with h5py.File(filename_h5oina, 'r+') as h5:
    h5['1/EBSD/Data/Phase'][:] = 0
    h5['1/EBSD/Data/Error'][:] = 2 # "No Solution"

