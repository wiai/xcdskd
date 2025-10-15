import h5py
import json
from hdf5_utils import load_dict_from_hdf5

filename_h5oina  = b"d:\\EBSD_Data\\maps\\MPI_DD\\Mn\\2019\\AD_0013_repolished\\05\\brkr\\AD_0013_repolished_EBSD05.h5oina"

h5oina_dict = load_dict_from_hdf5(filename_h5oina, exclude_list=['Processed Patterns'])

header = h5oina_dict['1']['EBSD']['Header']
map_width  = header['X Cells']
map_height = header['Y Cells']

print(map_width, map_height)
#with open("sample.json", "w") as outfile:
#    json.dump(h5oina_dict, outfile)


nap=1

