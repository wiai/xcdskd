import FreeSimpleGUI as sg
sg.theme('Dark Blue 3') 

import shutil
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread

# add aloe path and import aloe modules
import sys, os
aloe_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+"/src/"
aloe_h5oina_assets = os.path.join(aloe_root, "aloe/io/h5oina/assets")
sys.path.insert(0, aloe_root)
from aloe.io.edax import ang
from aloe.image.kikufilter import process_pattern_tsl
from aloe.image.downsample import downsample
from aloe.image.utils import img_to_uint
print(f"tsl2h5oina loaded aloe from: {aloe_root}")

# -------------------------------------------------------------------------------------------------
# GUI 
# -------------------------------------------------------------------------------------------------
# --- Input filename ----
fname = sg.popup_get_file('Input file: Select ANG file of the measurement',
    file_types=(("EDAX *.ang Files", "*.ang"),))
if not fname:
    sg.popup("Cancel", "No filename supplied")
    raise SystemExit("Cancelling: no filename supplied")

tsl_ang_file = fname
filename_base, file_extension = os.path.splitext(fname)

ang_header, ang_data = ang.read_ang_file(tsl_ang_file);
map_ncols = ang_header['NCOLS_EVEN']
map_nrows = ang_header['NROWS']
tsl_microns = ang_header['XSTEP']
tsl_xstar = ang_header['x-star']
tsl_ystar = ang_header['y-star']
tsl_zstar = ang_header['z-star'] 

tsl_sample_tilt = ang_header['SampleTiltAngle']
tsl_detector_tilt = ang_header['CameraElevationAngle']
print(f"\nSearching for pattern directory of {tsl_ang_file} ...")
tsl_pattern_dir = ang.find_grid_directory(tsl_ang_file)
extension, nrows, ncols, files, image_info = ang.organize_grid_files(tsl_pattern_dir)

if image_info:
    pattern_format = image_info[0]
    image_size = image_info[1]
    pattern_nrows = image_size[0]
    pattern_ncols = image_size[1]
    n_map_points = map_ncols * map_nrows
else:
    print("ERROR: Could not determine pattern image information!")
    sys.exit(1)

"""
select SEM kV of patterns
"""

layout = [  [sg.Text('File: ' + fname)],
            [sg.Combo(values = ['10','15','20','25','30'], enable_events=True, key='kvcombo', default_value='20'), sg.Text('kV')],
            [[sg.Button('Submit'), sg.Button('Cancel')]]
            ]
            
window = sg.Window('SEM Beam Voltage of EBSD measurment', layout)

process = False
while True:
    event, values = window.read()
    
    if event in (sg.WIN_CLOSED, 'Cancel'):
        break
    if event == 'Submit':
        process = True
        tsl_kV = int(values['kvcombo'])
        break
            
window.close()
if not process:
    raise SystemExit("Cancelled")

"""
select binning factor
"""

layout = [  [sg.Text('File: ' + fname)],
            [sg.Text('Unbinned Pattern Height: ' + str(pattern_nrows))],
            [sg.Text('Unbinned Pattern Width : ' + str(pattern_ncols))],
            [sg.Combo(values = ['1','2','3','4','5','6','7','8','11','12','14','16'], enable_events=True, key='bincombo', default_value='1')],
            [[sg.Button('Submit'), sg.Button('Cancel')]]
            ]
            
window = sg.Window('Resized Pattern Width', layout)

process = False
while True:
    event, values = window.read()
    
    if event in (sg.WIN_CLOSED, 'Cancel'):
        break
    if event == 'Submit':
        process = True
        binning = int(values['bincombo'])
        break
            
window.close()
if not process:
    raise SystemExit("Cancelled")

source_img = np.zeros([pattern_nrows, pattern_ncols])
binned_image = downsample(source_img, binning)
nrows_binned = binned_image.shape[0]
ncols_binned = binned_image.shape[1]

filename_h5oina_out = filename_base +"_"+str(tsl_kV)+"kV_"+str(nrows_binned)+"x"+str(ncols_binned)+".h5oina"

""" output filename
"""
layout = [[sg.Text('Enter Output Filename')],
                 [sg.InputText(filename_h5oina_out), sg.FileBrowse()],
                 [sg.Submit(), sg.Cancel()]]

window = sg.Window('OUTPUT FILE', layout)

event, values = window.read()
window.close()

if values[0]:
    filename_h5oina_out = values[0]     # the first input element is values[0]
else:
    raise SystemExit("Cancelled")
# -------------------------------------------------------------------------------------------------
# GUI END
# -------------------------------------------------------------------------------------------------



print(f"\nANG file: {tsl_ang_file}")
print(f"✓ Map  size (rows x cols): {map_nrows} x {map_ncols}")
print(f"✓ Step size     (microns): {tsl_microns}")

print(f"\nPattern data folder: {tsl_pattern_dir}")
print(f"✓ Extension: {extension}")
print(f"✓ Grid size: {nrows} rows × {ncols} columns")
print(f"✓ Found {len(files)} files")
print(f"✓ Image format: {pattern_format}")
print(f"✓ Pattern Size (rows x cols): {pattern_nrows} x  {pattern_ncols}")

#print(f"\nFiles organized by position:")
#print(f"{'Row Idx':<10} {'Col Idx':<10} {'Row Val':<10} {'Col Val':<10} {'Filename'}")
#print("-" * 80)
#for row_idx, col_idx, row_val, col_val, filename in files:
#    print(f"{row_idx:<10} {col_idx:<10} {row_val:<10} {col_val:<10} {filename}")
        
shutil.copyfile(os.path.join(aloe_h5oina_assets,"H5OINA_template.h5oina"), filename_h5oina_out)
h5oina_header = "1/EBSD/Header"
h5oina_data = "1/EBSD/Data"
h5oina_dataset_patterns = "1/EBSD/Data/Processed Patterns"

fd = h5py.File(filename_h5oina_out, 'r+')

#del fd["Proprietary/Settings"]
#del fd["Proprietary/Thumbnail"]
#del fd["Proprietary"]

#fd["Index"] = 1
#fd["Manufacturer"] = "Oxford Instruments"
#fd["Software Version"] = "6.0.7878.1"
#fd["Format Version"] = 4.0

fd[h5oina_header + "/X Step"][:] = tsl_microns
fd[h5oina_header + "/Y Step"][:] = tsl_microns
fd[h5oina_header + "/X Cells"][:] = map_ncols
fd[h5oina_header + "/Y Cells"][:] = map_nrows
fd[h5oina_header + "/Beam Voltage"][:] = tsl_kV

# Detector Orientation, vertical=90°
detector_XROT =  90.0 + tsl_detector_tilt
fd[h5oina_header + "/Detector Orientation Euler"][:] = np.radians(np.array([[0,detector_XROT,0]], dtype=np.float32))
fd[h5oina_header + "/Tilt Angle"][:] = np.radians(tsl_sample_tilt)

# copy binned patterns
nrows_binned = pattern_nrows // binning
ncols_binned = pattern_ncols // binning
print(f"\nProcessing and saving patterns (height x width): {nrows_binned} x {ncols_binned}")
print(f"Output filename: {filename_h5oina_out}")

fd["1/EBSD/Header/Pattern Height"][:] = nrows_binned
fd["1/EBSD/Header/Pattern Width"][:] = ncols_binned
fd.create_dataset(h5oina_dataset_patterns, 
    (n_map_points, nrows_binned, ncols_binned), 
    dtype=np.uint8, 
    chunks=(1, nrows_binned, nrows_binned),
    compression='gzip',compression_opts=9)


# Euler angles in rad
h5oina_euler = h5oina_data + "/Euler"
del fd[h5oina_euler]
fd.create_dataset(h5oina_euler, 
    (n_map_points, 3), 
    dtype=np.float32)

h5oina_x = h5oina_data + "/X"
del fd[h5oina_x]
fd.create_dataset(h5oina_x, 
    (n_map_points, 1), 
    dtype=np.float32)

h5oina_y = h5oina_data + "/Y"
del fd[h5oina_y]
fd.create_dataset(h5oina_y, 
    (n_map_points, 1), 
    dtype=np.float32)
    
h5oina_phase = h5oina_data + "/Phase"
del fd[h5oina_phase]
fd.create_dataset(h5oina_phase, data = np.zeros((n_map_points, 1), dtype=np.int32))
    
h5oina_BC = h5oina_data + "/Band Contrast"
del fd[h5oina_BC]
fd.create_dataset(h5oina_BC, 
    (n_map_points, 1), 
    dtype=np.int32)

h5oina_BS = h5oina_data + "/Band Slope"
del fd[h5oina_BS]
fd.create_dataset(h5oina_BS, 
    (n_map_points, 1), 
    dtype=np.int32)
    
h5oina_Bands = h5oina_data + "/Bands"
del fd[h5oina_Bands]
fd.create_dataset(h5oina_Bands, 
    (n_map_points, 1), 
    dtype=np.int32)
    
h5oina_PCX = h5oina_data + "/Pattern Center X"
del fd[h5oina_PCX]
fd.create_dataset(h5oina_PCX, 
    (n_map_points, 1), 
    dtype=np.float32)
    
h5oina_PCY = h5oina_data + "/Pattern Center Y"
del fd[h5oina_PCY]
fd.create_dataset(h5oina_PCY, 
    (n_map_points, 1), 
    dtype=np.float32)
    
h5oina_DD = h5oina_data + "/Detector Distance"
del fd[h5oina_DD]
fd.create_dataset(h5oina_DD, 
    (n_map_points, 1), 
    dtype=np.float32)
    
h5oina_PQ = h5oina_data + "/Pattern Quality"
del fd[h5oina_PQ ]
fd.create_dataset(h5oina_PQ , 
    (n_map_points, 1), 
    dtype=np.float32) 
    
h5oina_MAD = h5oina_data + "/Mean Angular Deviation"
del fd[h5oina_MAD ]
fd.create_dataset(h5oina_MAD , data = np.zeros((n_map_points, 1), dtype=np.float32))
      
h5oina_ERROR = h5oina_data + "/Error"
del fd[h5oina_ERROR]
fd.create_dataset(h5oina_ERROR, 
    (n_map_points, 1), 
    dtype=np.int32)      

# delete all phases except 2
try:
    for i in range(3,20):
        del fd[h5oina_header + "/Phases/"+str(i)]
except:
    pass

fd[h5oina_header + "/Phases/1/Color"][...] = np.array([[255, 0, 0]], dtype=np.uint8)   
fd[h5oina_header + "/Phases/1/Database Id"][...] = 0         
fd[h5oina_header + "/Phases/1/Lattice Angles"][...] = np.array([[1.5707964, 1.5707964, 1.5707964]], dtype=np.float32)      
fd[h5oina_header + "/Phases/1/Lattice Dimensions"][...] = np.array([[3.566, 3.566, 3.566]], dtype=np.float32)
fd[h5oina_header + "/Phases/1/Laue Group"][...] = 11  
fd[h5oina_header + "/Phases/1/Number Reflectors"][...] = 190 
fd[h5oina_header + "/Phases/1/Phase Id"][...] =  1 
fd[h5oina_header + "/Phases/1/Phase Name"][...] = "fcc"  
fd[h5oina_header + "/Phases/1/Reference"][...] = "unknown"  
fd[h5oina_header + "/Phases/1/Space Group"][...] = 225


fd[h5oina_header + "/Phases/2/Color"][...] = np.array([[0, 0, 255]], dtype=np.uint8)   
fd[h5oina_header + "/Phases/2/Database Id"][...] = 0         
fd[h5oina_header + "/Phases/2/Lattice Angles"][...] = np.array([[1.5707964, 1.5707964, 1.5707964]], dtype=np.float32)      
fd[h5oina_header + "/Phases/2/Lattice Dimensions"][...] = np.array([[2.86, 2.86, 2.86]], dtype=np.float32)
fd[h5oina_header + "/Phases/2/Laue Group"][...] = 11  
fd[h5oina_header + "/Phases/2/Number Reflectors"][...] = 190 
fd[h5oina_header + "/Phases/2/Phase Id"][...] =  2 
fd[h5oina_header + "/Phases/2/Phase Name"][...] = "bcc"  
fd[h5oina_header + "/Phases/2/Reference"][...] = "unknown"  
fd[h5oina_header + "/Phases/2/Space Group"][...] = 229


# set TSL mask, radius 0.47
boundaries = np.array([[0.03],[0.03],[0.94],[0.94]], dtype=np.float32)
fd['1/Data Processing/Pattern Matching/Current Settings/Pattern Weighting Ellipse'] = boundaries
fd['1/Data Processing/Pattern Matching/Current Settings/Pattern Weighting Mode'] = np.array(1, dtype=np.uint8)


for pattern_data in tqdm(files):
        
    irow = pattern_data[0]
    icol = pattern_data[1]
    h5oina_idx = icol + irow * map_ncols
    filename_pattern = os.path.join(tsl_pattern_dir, pattern_data[4])

    source_img = imread(filename_pattern, as_gray=True)
   
    if binning>1:
        source_img = downsample(source_img, int(binning))

    #processed_pattern = img_to_uint(source_img, dtype=np.uint8)
    
    processed_pattern = process_pattern_tsl(source_img, scale=None, sigma=0.08, rmax=0.47, 
        dtype=np.uint8, clow=0.2, chigh=99.8, static_background=None)

    #plt.figure()
    #plt.imshow(processed_pattern)
    #plt.show()
    
    fd[h5oina_dataset_patterns][h5oina_idx,:,:] = processed_pattern[:,:]
    
    # init with zero solutions
    fd[h5oina_phase][h5oina_idx,:] = 0

    fd[h5oina_PCX][h5oina_idx,:] = tsl_xstar
    fd[h5oina_PCY][h5oina_idx,:] = tsl_ystar
    fd[h5oina_DD][h5oina_idx,:]  = tsl_zstar
    
    fd[h5oina_BC][h5oina_idx,:] = np.mean(processed_pattern)
    fd[h5oina_BS][h5oina_idx,:] = np.std(processed_pattern)
    
    fd[h5oina_ERROR][h5oina_idx,:] = 1

fd.close()