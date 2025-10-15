import h5py
import numpy as np

'''
def unpack_rgba_from_1d(input_file, input_dataset, 
                       output_file, output_dataset,
                       nrows, ncols):
    """
    Unpack 1D 32-bit integers (RGBA packed) into 2D RGBA image and save to HDF5.
    
    Parameters:
    - input_file: Path to input HDF5 file
    - input_dataset: Name of 1D dataset in input file
    - output_file: Path to output HDF5 file
    - output_dataset: Name of image dataset in output file
    - nrows: Number of rows in output image
    - ncols: Number of columns in output image
    """
    
    # Open input file and read 1D dataset
    with h5py.File(input_file, 'r') as f_in:
        data_1d = f_in[input_dataset][:]  # Should be uint32 or int32
    
    # Verify dimensions
    if len(data_1d) != nrows * ncols:
        raise ValueError(f"1D dataset size ({len(data_1d)}) doesn't match nrows*ncols ({nrows*ncols})")
    
    # Reshape to 2D first (still packed RGBA)
    packed_2d = data_1d.reshape(nrows, ncols)
    
    # Unpack into separate channels (using numpy's dtype manipulation)
    # Method 1: Using view as uint8 (most efficient)
    rgba = packed_2d.view(np.uint8).reshape(nrows, ncols, 4)
    
    # For HDF5/HDFView compatibility, we might need to reverse the channel order
    # depending on how the data was packed (ARGB vs RGBA, etc.)
    # Here we assume RGBA order (byte order: [R,G,B,A])
    # If your data is stored as ARGB or BGRA, adjust accordingly:
    # rgba = rgba[..., [3,0,1,2]]  # For ARGB to RGBA
    # rgba = rgba[..., [2,1,0,3]]   # For BGRA to RGBA
    
    # Create output file with proper attributes for HDFView
    with h5py.File(output_file, 'w') as f_out:
        # Create dataset with image data
        dset = f_out.create_dataset(output_dataset, data=rgba)
        
        # Add attributes that HDFView uses to recognize this as an image
        dset.attrs['CLASS'] = np.string_('IMAGE')
        dset.attrs['IMAGE_VERSION'] = np.string_('1.2')
        dset.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
        dset.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')
        dset.attrs['IMAGE_MINMAXRANGE'] = np.array([0, 255], dtype=np.uint8)
'''

def unpack_rgba_to_same_file(input_file, input_dataset, output_dataset, nrows, ncols):
    """
    Unpack 1D 32-bit integers (RGBA packed) into 2D RGBA image and save in the same HDF5 file.
    
    Parameters:
    - input_file: Path to input HDF5 file (will be modified in-place)
    - input_dataset: Name of 1D dataset in input file
    - output_dataset: Name of new image dataset to create
    - nrows: Number of rows in output image
    - ncols: Number of columns in output image
    """
    
    # Open input file in read/write mode
    with h5py.File(input_file, 'r+') as f:
        # Read the 1D dataset
        data_1d = f[input_dataset][:]  # Should be uint32 or int32
        
        # Verify dimensions
        if len(data_1d) != nrows * ncols:
            raise ValueError(f"1D dataset size ({len(data_1d)}) doesn't match nrows*ncols ({nrows*ncols})")
        
        # Reshape to 2D and unpack into RGBA
        packed_2d = data_1d.reshape(nrows, ncols)
        rgba = packed_2d.view(np.uint8).reshape(nrows, ncols, 4)
        
        # Create or overwrite the output dataset
        if output_dataset in f:
            del f[output_dataset]  # Delete if already exists
        
        dset = f.create_dataset(output_dataset, data=rgba)
        
        # Add attributes for HDFView compatibility
        dset.attrs['CLASS'] = np.string_('IMAGE')
        dset.attrs['IMAGE_VERSION'] = np.string_('1.2')
        dset.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
        dset.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')
        dset.attrs['IMAGE_MINMAXRANGE'] = np.array([0, 255], dtype=np.uint8)

# Example usage
if __name__ == "__main__":
    # Input parameters - adjust these as needed
    input_filename = "G4N Memory 2 Slow Specimen 1 Zone Step_0_24.4DegC_Acquisition 8.h5oina"
    input_dset_name = "1/Data Processing/Results/Maps/3/Pixels"
    output_dset_name = "1/Data Processing/Results/Maps/3/RGBA_Bitmap"
    rows = 798  # Adjust based on your image dimensions
    cols = 799  # Adjust based on your image dimensions
    
    unpack_rgba_to_same_file(input_filename, input_dset_name,
                       output_dset_name, rows, cols)
    
    #unpack_rgba_from_1d(input_filename, input_dset_name,
    #                   output_filename, output_dset_name,
    #                   rows, cols)