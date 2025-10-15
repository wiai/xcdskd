import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

import os
import numpy as np
import h5py
from tqdm import tqdm


def column_vector(x,y,z):
    """ 3x1 matrix from coordinate triple x,y,z
    Create an explicit 3x1 numpy column vector.
    """
    return np.array([[x,y,z]]).T


def lstr(*args):
    """string representation of argument list elements as one string
    use for logging
    """
    list_string = ' '.join(map(str, args)) 
    return list_string


def copy_datasets(filename_source, filename_target, 
    filter_strings=["/1/Data Processing", "Processed Patterns"]):
    """ copy hdf5 datsets, without datasets  containing strings in filter_strings
    """
    fs = h5py.File(filename_source, 'r')
    fd = h5py.File(filename_target, 'w')
    for a in fs.attrs:
        fd.attrs[a] = fs.attrs[a]
    
    def visitor_func(name, node):
        for nodefilter in filter_strings:
            if (nodefilter in node.name):
                # skip this dataset
                # print("Skipping: ", node.name)
                return
        
        if isinstance(node, h5py.Dataset):
            # node is a dataset
            #if (not datasetname in node.name):
            group_path = node.parent.name
            group_id = fd.require_group(group_path)
            #print(group_path)
            #print(node.name)
            fs.copy(node.name, group_id, name=node.name)

        #else:
        #     print("GROUP: ", node.keys())
        
        return
    
    fs.visititems(visitor_func)
    fs.close()
    fd.close()
    return


def add_laue_symbols(filename):
    """
    add the Laue group symbols as attributes in the phase list
    """
    laue_symbols = {}
    laue_symbols[1] = "-1"
    laue_symbols[2] = "2/m"
    laue_symbols[3] = "mmm"
    laue_symbols[4] = "4/m"
    laue_symbols[5] = "4/mmm"
    laue_symbols[6] = "-3"
    laue_symbols[7] = "-3m"
    laue_symbols[8] = "6/m"
    laue_symbols[9] = "6/mmm"
    laue_symbols[10] = "m-3"
    laue_symbols[11] = "m-3m"
    
    # string termination issues?
    # https://forum.hdfgroup.org/t/nullpad-nullterm-strings/9107    
    # use PyTables?
    # https://github.com/PyTables/PyTables/issues/264
    
    f = h5py.File(filename, 'a')
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            # node is a dataset
            if ('Laue Group' in node.name):
                g = int(f[node.name][()])
                symbol = laue_symbols.get(g, "n.d.")
                #symbol = symbol + '\x00'
                #length = len(symbol.encode("ascii"))
                #ascii_type = h5py.string_dtype('ascii', length)
                LOG.info(lstr('Setting Laue group symbol attribute: ', g, symbol))
                #f[node.name].attrs["Symbol"] = np.array(symbol.encode('ascii'), dtype=ascii_type)
                f[node.name].attrs["Symbol"] = symbol #symbol.encode('utf8')
            if ('Space Group' in node.name):
                g = int(f[node.name][()])
                symbol = laue_symbols.get(g, "---")
                #symbol = symbol + '\x00'
                #length = len(symbol.encode("ascii"))
                #ascii_type = h5py.string_dtype('ascii', length)
                LOG.info(lstr('Setting Space group symbol attribute: ', g, symbol))
                #f[node.name].attrs["Symbol"] = np.array(symbol.encode('ascii'), dtype=ascii_type)
                f[node.name].attrs["Symbol"] = symbol
        else:
            # node is a group
            return

    f.visititems(visitor_func)
    f.close()
    return


def get_pattern_dataset_shape(filename_h5oina):
    """ get pattern size from h5oina file
    """
    dataset_patterns = "1/EBSD/Data/Processed Patterns"
    fs = h5py.File(filename_h5oina, 'r')
    patterns = fs[dataset_patterns]
    n_patterns = patterns.shape[0]
    pattern_nrows = patterns.shape[1]
    pattern_ncols = patterns.shape[2]
    fs.close()
    return [n_patterns, pattern_nrows, pattern_ncols]
    
    
def get_map_shape(filename_h5oina):
    """ get pattern size from h5oina file
    """
    fs = h5py.File(filename_h5oina, 'r')
    map_nrows = np.ravel(fs["1/EBSD/Header/Y Cells"])[0]
    map_ncols = np.ravel(fs["1/EBSD/Header/X Cells"])[0]
    fs.close()
    return [map_nrows, map_ncols]


def bin_patterns(filename_source, filename_target, binning=1, filter_strings=["/1/Data Processing", "Processed Patterns"]):
    """ copy hdf5 to new file, with binned patterns
    """
    dataset_patterns = "1/EBSD/Data/Processed Patterns"
    nodefilter = "/1/Data Processing"
    
    fs = h5py.File(filename_source, 'r')
    fd = h5py.File(filename_target, 'w')
    for a in fs.attrs:
        fd.attrs[a] = fs.attrs[a]
    
    # pattern dataset should be npixels, pattern_nrows, pattern_ncols 
    patterns = fs[dataset_patterns]
    print("Unbinned pattern dataset shape:", patterns.shape)
    n_patterns = patterns.shape[0]
    pattern_nrows = patterns.shape[1]
    pattern_ncols = patterns.shape[2]

    def visitor_func(name, node):
        for nodefilter in filter_strings:
            if (nodefilter in node.name):
                #print("Skipping: ", node.name)
                return
        
        if isinstance(node, h5py.Dataset):
            group_path = node.parent.name
            group_id = fd.require_group(group_path)
            #print(group_path)
            #print(node.name)
            #fs.copy(node.name, fd)
            fs.copy(node.name, group_id, name=node.name)

        #else:
        #     print("GROUP: ", node.keys())
        return
    
    # copy all datasets from fs to fd
    fs.visititems(visitor_func)
    
    # binned patterns, check final size
    source_img = fs[dataset_patterns][0]
    binned_image = downsample(source_img, binning).astype(np.uint8)
    nrows_binned = binned_image.shape[0]
    ncols_binned = binned_image.shape[1]
    print("Binning Patterns to height,width: ", nrows_binned, ncols_binned)

    # bin the static background if exsisting
    dataset_bg = "1/EBSD/Header/Processed Static Background"
    if dataset_bg in fd:
        print("Binning the Static Background...")
        source_img = np.copy(fd[dataset_bg][()])
        del fd[dataset_bg]
        binned_image = img_to_uint(downsample(source_img, binning), clow=0.5, chigh=99.5, dtype=np.uint8).astype(np.uint8)
        fd[dataset_bg] = binned_image


    try:
        del fd["1/EBSD/Header/Pattern Height"]
        del fd["1/EBSD/Header/Pattern Width"]
    except:
        pass
    fd["1/EBSD/Header/Pattern Height"] = nrows_binned
    fd["1/EBSD/Header/Pattern Width"] = ncols_binned
    

    fd.create_dataset(dataset_patterns, 
        (n_patterns, nrows_binned, ncols_binned), 
        dtype=np.uint8, 
        chunks=(1, nrows_binned, ncols_binned),
        compression='gzip',compression_opts=9)

    for i in tqdm(range(n_patterns)):
        binned_image = np.zeros((nrows_binned, ncols_binned), dtype=np.uint8)
        try:
            source_img = fs[dataset_patterns][i] 
            #binned_image = downsample(source_img, binning).astype(np.uint8)
            binned_image = img_to_uint(downsample(source_img, binning), clow=0.5, chigh=99.5, dtype=np.uint8).astype(np.uint8)
        except:
            pass
            
        fd[dataset_patterns][i,:,:] = binned_image[:,:]
    
    fd.close()
    fs.close()

    return
    
    

###############################################################################
# image utility functions
###############################################################################
from skimage import exposure

def img_standardize(img):
    """standardize image to have mean=0 and stddev=1
    """
    mean_img = np.mean(img)
    std_img = np.std(img)
    return (img-mean_img)/(std_img) 


def img_to_uint(img, clow=0.25, chigh=99.75, dtype=np.int8):
    """ convert a numpy array to unsigned integer 8/16bit
    stretch contrast to include (clow..chigh) percent of original intensity
    
    Note:
    skimage.img_as_ubyte etc can be sensitive to outliers as scaling is to min/max
    
    saving:
    skimage.io.imsave("name.png", img_to_uint(array))
    
    16bit example:
    skimage.io.imsave('img_corrected_16bit.tif', 
    img_to_uint(array, to16bit=True), plugin='tifffile')
        
    """
    # set maximum integer value
    #if to16bit:
    #    maxint=(2**16 -1)
    #else:
    #    maxint=(2**8  -1)
        
    # get percentiles    
    p_low, p_high = np.percentile(img, (clow, chigh)) # clow, chigh in percent 0...100
    img_rescaled = exposure.rescale_intensity(img, in_range=(p_low, p_high), out_range=dtype)
    img_uint=img_rescaled.astype(dtype)    
    
    # image range for 0..maxint of uint data type
    """
    vmin=np.min(img_rescale)
    vmax=np.max(img_rescale)
    
    # scale and clip to uint range
    img_int=maxint * (img_rescale-vmin)/(vmax-vmin)
    img_int=np.clip(img_int, 1, maxint)
    
    # change to unsigned integer data type 
    if to16bit:
        img_uint=img_int.astype(np.uint16)
    else:
        img_uint=img_int.astype(np.uint8)
    """
    return img_uint 


def img_uint_std(img, onemask, std=4.0, dtype=np.uint16):
    """ make 8/16bit image scaled to mean=0, stdev +/- std
    """

    # make sure that image has only positive values
    img_positive = np.copy(img) 
    img_positive -= np.min(img_positive) 
    img_positive += 1.0
        
    img_masked = img_positive * onemask
    
    img_norm, npix = norm_img_mask(img_masked)
    img_std = img_norm * np.sqrt(npix)
    img_clipped = np.clip(img_std, -std, +std)
        
    img_uint = img_to_uint(img_clipped, clow=0.0, chigh=100.0, dtype=dtype)
        
    return img_uint


def img_to_signed_int(img, std_low=-5.0, std_high=5.0, dtype=np.int8):
    """ convert a numpy array to signed integer 8/16bit
    """
    img_rescaled = exposure.rescale_intensity(img, in_range=(std_low, std_high), out_range=dtype)
    img_int = img_rescaled.astype(dtype)    
    return img_int 


def img_range(img, clow=1.0, chigh=99.0):
    """ get percentiles of image histogram
        for adjusting the color range limits in plots
    """    
    p_low, p_high = np.percentile(img, (clow, chigh))
    return [p_low, p_high]
    
    
def circular_mask(h, w, center=None, rmax=0.5):
    """ Return weight array of circular shape.
    
    Weight values of the mask array outside of rmax are 0, inside rmax the values are 1.
    Multiplication by mask sets masked,unwanted pixels to zero.
    """
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
        
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.hypot((X - center[0]), (Y-center[1]))
    onezeromask = dist_from_center <= rmax * w
    
    return onezeromask.astype(np.int)
    
    
"""
Downsample a numpy array. Use for binning of images.

This code is (c) Adam Ginsburg (agpy)

Image Tools from:
https://github.com/keflavich/image_tools

https://github.com/keflavich/image_tools/blob/master/image_tools/downsample.py
"""

def downsample(myarr,factor,estimator=np.nanmean):
    """
    Downsample a 2D array by averaging over *factor* pixels in each axis.
    Crops upper edge if the shape is not a multiple of factor.

    This code is pure np and should be fast.

    keywords:
        estimator - default to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
    """
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    return dsarr


def downsample_cube(myarr,factor,ignoredim=0):
    """
    Downsample a 3D array by averaging over *factor* pixels on the last two
    axes.
    """
    if ignoredim > 0: myarr = myarr.swapaxes(0,ignoredim)
    zs,ys,xs = myarr.shape
    crarr = myarr[:,:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = mean(np.concatenate([[crarr[:,i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    if ignoredim > 0: dsarr = dsarr.swapaxes(0,ignoredim)
    return dsarr


def downsample_1d(myarr,factor,estimator=np.nanmean):
    """
    Downsample a 1D array by averaging over *factor* pixels.
    Crops right side if the shape is not a multiple of factor.

    This code is pure np and should be fast.

    keywords:
        estimator - default to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
    """
    xs = myarr.shape
    crarr = myarr[:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor] 
        for i in range(factor)] ]),axis=0)
    return dsarr


def downsample_axis(myarr, factor, axis, estimator=np.nanmean, truncate=False):
    """
    Downsample an ND array by averaging over *factor* pixels along an axis.
    Crops right side if the shape is not a multiple of factor.

    This code is pure np and should be fast.

    keywords:
        estimator - default to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
    """
    # size of the dimension of interest
    xs = myarr.shape[axis]
    
    if xs % int(factor) != 0:
        if truncate:
            view = [slice(None) for ii in range(myarr.ndim)]
            view[axis] = slice(None,xs-(xs % int(factor)))
            crarr = myarr[view]
        else:
            newshape = list(myarr.shape)
            newshape[axis] = (factor - xs % int(factor))
            extension = np.empty(newshape) * np.nan
            crarr = np.concatenate((myarr,extension), axis=axis)
    else:
        crarr = myarr

    def makeslice(startpoint,axis=axis,step=factor):
        # make empty slices
        view = [slice(None) for ii in range(myarr.ndim)]
        # then fill the appropriate slice
        view[axis] = slice(startpoint,None,step)
        return view

    # The extra braces here are crucial: We're adding an extra dimension so we
    # can average across it!
    stacked_array = np.concatenate([[crarr[makeslice(ii)]] for ii in range(factor)])

    dsarr = estimator(stacked_array, axis=0)
    return dsarr


def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))


def load_dict_from_hdf5(filename, exclude_list=[]):
    """
    loads a HDF5 file into a dictionary, 
    negelects items (Datasets, groups) that contain strings in exclude_list
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/', exclude_list=exclude_list)


def recursively_load_dict_contents_from_group(h5file, path, exclude_list=[]):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        is_excluded = False
        for exclude in exclude_list:
            if exclude in key:
                is_excluded=True
        if not is_excluded:
            if isinstance(item, h5py._hl.dataset.Dataset):
                entry = item[()]
                ans[key] = entry
                #if hasattr(entry, 'tolist'):
                #    ans[key] = entry.tolist()
                #else:
                #    ans[key] = entry
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', exclude_list=exclude_list)
    return ans


if __name__ == '__main__':
    """
    data = {'x': 'astring',
            'y': np.arange(10),
            'd': {'z': np.ones((2,3)),
                  'b': b'bytestring'}}
    print(data)
    filename = 'test.h5'
    save_dict_to_hdf5(data, filename)
    dd = load_dict_from_hdf5(filename)
    print(dd)
    # should test for bad type
    """









