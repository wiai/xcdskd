""" Filtering routines for Kikuchi patterns. """

import numpy as np
import numba
import math as m

from PIL import Image


from .filterfft import filter, gaussian
#from scipy.ndimage import gaussian_filter

from .downsample import downsample
from .utils import img_to_uint, img_to_signed_int
from .nxcc import mask_pattern_disk, standard_img


@numba.jit(nopython=True)
def lmsd(img, kernelsize, result):
    h = img.shape[0]
    w = img.shape[1]
    
    nkernelpoints = (2*kernelsize+1)**2
    for i in range(0,h):
        for j in range(0,w):
            sum = 0.0
            for ik in range(-kernelsize, kernelsize+1):
                for jk in range(-kernelsize, kernelsize+1):
                    
                    ip = i + ik
                    jp = j + jk 
                    if (ip<0):
                        ip *= -1
                    if (ip>(h-1)):
                        ip = (h-1) - (ip-h)
                    if (jp<0):
                        jp *= -1
                    if (jp>(w-1)):
                        jp = (w-1) - (jp-w)
                    
                    sum += img[ip,jp]
            mean = sum/nkernelpoints
    
            # stddev
            ssum = 0.0
            for ik in range(-kernelsize, kernelsize+1):
                for jk in range(-kernelsize, kernelsize+1):
                    
                    ip = i + ik
                    jp = j + jk 
                    if (ip<0):
                        ip *= -1
                    if (ip>(h-1)):
                        ip = (h-1) - (ip-h)
                    if (jp<0):
                        jp *= -1
                    if (jp>(w-1)):
                        jp = (w-1) - (jp-w)
                    
                    ssum += (img[ip,jp]-mean) * (img[ip,jp]-mean)
            stddev = m.sqrt(ssum/nkernelpoints)

            result[i,j] = 0.0
            if (abs(stddev) > 1e-8):
                result[i,j] = (img[i,j] - mean)/stddev
            
    return


def filter_lmsd(img, kernelsize_pix):
    img_filtered = np.zeros_like(img)
    lmsd(img, kernelsize_pix, img_filtered)
    return img_filtered


def imresize(arr, size, resample=0):
    resized = np.array(Image.fromarray(arr).resize(size, resample=resample))
    return resized


def create_circular_mask(h, w, center=None, rmax=0.5):
    """ makes a weight array 
    values outside of rmax are 0 / inside rmax=1
    """
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
        
    Y, X = np.ogrid[:h, :w]
    #dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    dist_from_center = np.hypot((X - center[0]), (Y-center[1]))
    mask = dist_from_center <= rmax * w
    return mask


def mask_image(image, mask=None):
    if mask is None:
        mask = create_circular_mask(*image.shape)
    image_masked = image * mask
    return image_masked
    
    
def make_signal_bg_fft(image, sigma=None, support=None, bgscale=1.0):
    """ 
    Partition the _image_ into a large-scale smooth _background_ 
    and a small-scale _signal_, 
    both add up to original _image_
    
    The spatial low-frequency background is estimated 
    by a gaussian convolution FFT filter
        
    Parameters:

    sigma   : sigma of Gaussian, defaults to 1/25 of image width
    support : extension of Kernel, should be a few sigmas
    bgscale : scale the background, use e.g. to ensure that signal>=0
    
    """
    
    if sigma is None:
        sigma=image.shape[1]/25
    
    if support is None:
        support=(3*sigma, 3*sigma)
    
    background = bgscale*filter(image, gaussian(sigma,support))
    #background = bgscale*gaussian_filter(image, sigma)
    
    signal=image-background
    return signal, background
    
BGK = make_signal_bg_fft
    
    
def remove_bg_fft(image, sigma=None, support=None):
    """ 
    Divide image by dynamic background from fft low pass filter.
    
        
    Parameters:

    sigma   : sigma of Gaussian
    support : extension of Kernel, should be a few sigmas
    bgscale : scale FFT determined background, use e.g. to ensure that signal>=0
    
    """
    if sigma is None:
        sigma=image.shape[0]/8
    
    if support is None:
        support=(3*sigma, 3*sigma)
    
    dynamic_bg = filter(image, gaussian(sigma,support))
    
    img = np.true_divide(image, dynamic_bg)
    img = np.nan_to_num(img, nan=0.0)
    #img = np.true_divide(image, gaussian_filter(image, sigma))
    return img

    
@numba.jit(nopython=True)
def fill_pattern_tsl(pattern, rmax=0.47):
    """ fill the outer part of tsl pattern by inversion at circle """
    h, w = pattern.shape
    xc = w // 2
    yc = h // 2
    pattern_filled = np.copy(pattern)
    rmax_px = w * rmax
    for iy in range(h):
        dy = (iy-yc)
        for ix in range(w):
            dx = (ix-xc)
            r = np.sqrt(dy**2 + dx**2)
            if (r>rmax_px):
                rinv = rmax_px/r
                xinv = int(xc + dx*rinv)
                yinv = int(yc + dy*rinv)
                pattern_filled[iy,ix] = pattern[yinv,xinv]
                
    return pattern_filled



def process_pattern_tsl(experiment, scale=None, sigma=0.05, rmax=0.47, 
    dtype=np.uint16, clow=0.01, chigh=99.99, 
    static_background=None):
    """ process experimental TSL pattern, rescale, mask by circle 
    """
    h, w = experiment.shape
    if scale is not None:
        experiment = imresize(experiment, (int(h*scale), int(w*scale)))
        if static_background is not None:
            static_background = imresize(static_background, (int(h*scale), int(w*scale)))
        
    if static_background is not None:
        experiment = np.true_divide(experiment, static_background)
 
    experiment = fill_pattern_tsl(experiment, rmax=rmax)
    
    experiment = remove_bg_fft(experiment, sigma=w*sigma)
    
    # lmsd filter
    kernelsize_pix = int(0.5 * sigma * w)
    experiment = filter_lmsd(experiment, kernelsize_pix)
    
    # conversion to 8 or 16 bit integer data
    if (dtype is np.uint16) or (dtype is np.uint8):
        experiment = img_to_uint(experiment, clow=clow, chigh=chigh, dtype=dtype)
    
    experiment, points = mask_pattern_disk(experiment, rmax=rmax)
    
    return experiment


def process_ebsp(raw_pattern=None, static_background=None, sigma=None,
                 clow=0.01, chigh=99.99, dtype=np.uint8, binning=1, lmsd=None):
    """
    Basic image processing for raw EBSD pattern data.
    
    Notes:

    Extra pattern binning is applied to the raw_pattern and the static_background.
    For more speed in mapping etc, consider having the raw_pattern
    and static_background already binned and set binning=1 here, to avoid
    the repeated binning of the static_background. 
    """
    # optional binning
    if binning>1:
        pattern_binned = downsample(raw_pattern, binning)
        if not (static_background is None):
            static_background_binned = downsample(static_background, binning)
    else:
        pattern_binned = raw_pattern
        static_background_binned = static_background
        
    # static background correction
    if static_background is None:
        pattern_static=pattern_binned
    else:
        pattern_static=np.true_divide(pattern_binned, static_background_binned)
    
    # dynamic background correction
    sigma_pix = sigma # sigma is given in pixels
    pattern_dyn = remove_bg_fft(pattern_static, sigma=sigma_pix)
    
    # lmsd filter
    if lmsd is not None:
        kernelsize_pix = lmsd_rel * raw_pattern.shape[1]
        pattern_dyn = filter_lmsd(pattern_dyn, kernelsize_pix)
    
    # conversion to 8 or 16 bit integer data
    if (dtype == np.uint8) or (dtype == np.uint16): 
        pattern_result = img_to_uint(pattern_dyn, clow=clow, chigh=chigh, dtype=dtype)
    else:
        pattern_result = pattern_dyn

    return pattern_result    
    
    
