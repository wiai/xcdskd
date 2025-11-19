# add tiff patterns to h5oina file, tiff contain PC
filename_h5oina = "d:/EBSD_Data/Steel crack HR EBSD/mapdata_aztec/Unprocessed 1/Steel_Crack.h5oina"
filename_patterns =r"d:/EBSD_Data/Steel crack HR EBSD/mapdata_aztec/Unprocessed 1/Images/"
filename_bg = r"d:\EBSD_Data\Steel crack HR EBSD\mapdata_aztec\Unprocessed 1\StaticBackground\StaticBackground.tiff"
BINNING=2

import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread

# add aloe path and import aloe modules
import sys, os
aloe_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+"/src/"
aloe_h5oina_assets = os.path.join(aloe_root, "aloe/io/h5oina/assets")
sys.path.insert(0, aloe_root)
from aloe.io.edax import ang
from aloe.io.h5oina.oina_tiff import get_oina_tiff_pc_from_file
from aloe.image.kikufilter import process_ebsp
from aloe.image.downsample import downsample
from aloe.image.utils import img_to_uint
from aloe.plots import plot_image
print(f"loaded aloe from: {aloe_root}")

static_bg = imread(filename_bg, as_gray=True)

        
header_group = "/1/EBSD/Header"
pattern_dataset = "/1/EBSD/Data/Processed Patterns"
pcx_dataset = "/1/EBSD/Data/Pattern Center X"
pcy_dataset = "/1/EBSD/Data/Pattern Center Y"
dd_dataset = "/1/EBSD/Data/Detector Distance"

h5 = h5py.File(filename_h5oina, "r+") # r+ read/write, file must exist

map_nrows =  h5[header_group + "/Y Cells"][()][0]
map_ncols =  h5[header_group + "/X Cells"][()][0]
n_patterns = map_ncols * map_nrows

try:
    del h5[header_group + "/Pattern Height"] 
    del h5[header_group + "/Pattern Width"]
    del h5[pcx_dataset]
    del h5[pcy_dataset]
    del h5[dd_dataset]
except:
    pass

nzerofill = 0
pattern_nrows = 1024 // BINNING
pattern_ncols = 1244 // BINNING

h5[header_group + "/Pattern Height"] = pattern_nrows
h5[header_group + "/Pattern Width"] = pattern_ncols

print("Map     rows, cols :", map_nrows, " ", map_ncols)
print("Pattern rows, cols :", pattern_nrows, " ", pattern_ncols)

h5.require_dataset(pattern_dataset, (n_patterns, pattern_nrows, pattern_ncols),
                  dtype=np.uint8,
                  chunks=(1,pattern_nrows,pattern_ncols),
                  compression='gzip', compression_opts=9 )

h5[pcx_dataset] = np.zeros(n_patterns, dtype=float)
h5[pcy_dataset] = np.zeros(n_patterns, dtype=float)
h5[dd_dataset]  = np.zeros(n_patterns, dtype=float) 

for irow in tqdm(range(map_nrows)):
    for icol in range(map_ncols):
        idx = irow * map_ncols + icol
        
        #filename = filename_patterns + str(idx+1).zfill(nzerofill)+".tiff"
        filename = filename_patterns + str(irow)+'_'+str(icol)+".tiff"
        img = np.array(imread(filename, as_gray=True))

        #lmsd = 0.06
        img_processed = process_ebsp(img, binning=BINNING, static_background= static_bg, 
            clow=0.25, chigh=99.75, dtype=np.uint8, lmsd=None)

        #plot_image(img_processed)
        
        pcx, pcy, dd = get_oina_tiff_pc_from_file(filename)
        h5[pattern_dataset][idx,:,:] = img_processed[:,:]
        h5[pcx_dataset][idx] = pcx
        h5[pcy_dataset][idx] = pcy
        h5[dd_dataset][idx] = dd