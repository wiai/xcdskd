""" Filtering routines for Kikuchi patterns. """

import numpy as np
import numba

from .filterfft import filter, gaussian
from .downsample import downsample
from .utils import img_to_uint
from .nxcc import mask_pattern_disk


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
    
    return np.true_divide(image, filter(image, gaussian(sigma,support)))

    
@numba.jit(nopython=True)
def fill_pattern_tsl(pattern, rmax=0.48):
    """ fill the outer part of tsl pattern by inversion at circle """
    h, w = pattern.shape
    xc = w // 2
    yc = h // 2
    pattern_filled = np.copy(pattern)
    rmax_px = h * rmax
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


def process_pattern_tsl(experiment, scale=None, sigma=0.1, rmax=0.48, dtype=np.uint16):
    """ process experimental TSL pattern, rescale, mask by circle """
    h,w = experiment.shape
    if scale is not None:
        experiment = resize(experiment, (h*scale, w*scale))
    experiment = fill_pattern_tsl(experiment, rmax=rmax)    
    experiment = np.array(process_ebsp(experiment, sigma=w*sigma, dtype=dtype) + 0.001).astype(np.float32)
    experiment, points = mask_pattern_disk(experiment, rmax=rmax)
    return experiment
    
    
def process_ebsp(raw_pattern=None, static_background=None, sigma=None,
                 clow=0.5, chigh=99.5, dtype=np.uint8, binning=1):
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
    pattern_dyn=remove_bg_fft(pattern_static, sigma=sigma)
    
    # conversion to 8 or 16 bit integer data
    pattern_int=img_to_uint(pattern_dyn, clow=clow, chigh=chigh, dtype=dtype)
    
    return pattern_int    
    

    