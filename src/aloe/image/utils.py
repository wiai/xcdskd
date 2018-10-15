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


def img_to_uint(img, clow=0.25, chigh=99.75, dtype=np.uint8):
    """ convert a numpy array to unsigned integer 8/16bit
    stretch contrast to include (clow..chigh)*100 percent of original intensity
    
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
    p_low, p_high = np.percentile(img, (clow, chigh))
    img_rescale = exposure.rescale_intensity(img, in_range=(p_low, p_high), out_range=dtype)
    img_uint=img_rescale.astype(dtype)    
    
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


def img_range(img, clow=1.0, chigh=99.0):
    """ get percentiles of image histogram
        for adjusting the color range limits in plots
    """    
    p_low, p_high = np.percentile(img, (clow, chigh))
    return [p_low, p_high]