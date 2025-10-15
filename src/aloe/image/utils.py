"""
image utility functions
"""
import numpy as np
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