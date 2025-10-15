""" Power SPectrum Analysis of Kikuchi Patterns
"""


import numpy as np
import numpy.ma as ma
from numpy import fft

import numba

from scipy import ndimage
from skimage.io import imread
import matplotlib.pyplot as plt

from .kikufilter import create_circular_mask, mask_image

 
def norm_img_mask(img, mask=None):
    """ normalize image to have mean=0 and stddev=1
        take only positive definite pixel values
    """
    if mask is None:
        mimg = ma.masked_less_equal(img, 0)
    else:
        mimg = ma.masked_array(img, mask=mask)

    npixels = mimg.size-np.sum(mimg.mask)    
    mean_mimg = np.mean(mimg)
    std_mimg = np.std(mimg)
    mimg = (mimg - mean_mimg)/(std_mimg*np.sqrt(npixels))
    return mimg, npixels

def standard_img_mask(img, mask=None):
    """ normalize image to have mean=0 and stddev=1
        take only positive definite pixel values
    """
    stdimage, npixels = norm_img_mask(img, mask=mask)
    stdimage *= np.sqrt(npixels) 
    return stdimage


@numba.jit(nopython=True) 
def normalize_image_power(img):
    """ subtract mean and divide by standard deviation
    """
    mean_img = np.mean(img)
    std_img = np.std(img)
    nimg = (img - mean_img)/(std_img)
    return nimg


def power_spectrum_2d(img):
    """ calculate 2d power spectrum via FFT
    """
    img_fft = fft.fft2(img, norm="ortho")
    
    # shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    f2 = fft.fftshift(img_fft)
    
    # calculate the 2D power spectrum
    psd2D = np.abs(f2)**2
    return psd2D

    
def hanning2d(M, N):
    """
    Hanning window for 2D
    """
    return np.outer(np.hanning(M),np.hanning(N))

    
def kaiser2d(M, N,beta):
    """
    Kaiser window for 2D
    """
    return np.outer(np.kaiser(M,beta),np.kaiser(N,beta))  



def high_pass_gaussian2d(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    low_pass = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    
    return  1.0 - low_pass



def fft_filter(im, spectral_window):
    """ multiply power spectrum of image 'im' by spectral window and
        transform back to new image 'im_new'
        
        im and spectral window are assumed to have the same shape (h,w)
    """
    im_fft = np.fft.fftshift(np.fft.fft2(im))
    if spectral_window is None:
        spectral_window = np.ones_like(im)
    im_fft2 = spectral_window * im_fft.copy()
    im_new = np.fft.ifft2(np.fft.ifftshift(im_fft2)).real
    return im_new


def plot_power_spectrum_2d(pspec, limits=[5e2,1e6],
    filename=None):
    """ plot 2d power spectrum
    """
    
    from matplotlib.colors import LogNorm

    nr, nc = pspec.shape
    col_freq = np.fft.fftshift(np.fft.fftfreq(nc))
    row_freq = np.fft.fftshift(np.fft.fftfreq(nr))
    extent = [row_freq[0],row_freq[-1],row_freq[0],col_freq[-1]]
    # logarithmic colormap
    if limits is not None:
        plt.imshow(pspec, norm=LogNorm(vmin=limits[0], vmax=limits[1]),
               extent = extent,
               cmap="plasma")
    else:
        #let matplotlib make limits
        plt.imshow(pspec, norm=LogNorm(vmin=5),
               extent = extent, cmap="plasma")
    plt.colorbar()
    plt.xlabel("cycles/pixel")
    plt.ylabel("cycles/pixel")
    
    if filename is not None:
        plt.savefig(filename, dpi=300)
    
    return    


def make_radial_labels(img):
    """ make labels array with radius from center
    """
    h, w  = img.shape
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)
    return r




def radial_power_spectrum(psd2D):
    """ Get PSD 1D (total radial power spectrum)
    
        Source:
        https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
    """
    wc  = psd2D.shape[1]//2
    r = make_radial_labels(psd2D)
    
    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.mean(psd2D, r, index=np.arange(0, wc))
    #stddev = ndimage.standard_deviation(psd2D, r, index=np.arange(0, wc))
    freq = np.fft.fftshift(np.fft.fftfreq(2*wc))[0:wc]
    freq = -np.flip(freq)
    return freq, psd1D

    
    
def plot_radial_power_spectrum(freq, rps, 
    filename=None, limits=None, limits_spatial=None):
    """ plot of the radial power spectrum
    """
    fig,ax = plt.subplots(figsize=(4,3.5))
    fig.patch.set_alpha(1)
    plt.loglog(1.0/freq, rps, linestyle="None", 
        marker="+", markersize=4, 
        alpha=0.5, color="k")
    if limits is not None:
        plt.ylim(*limits)
    if limits_spatial is not None:
        plt.xlim(*limits_spatial)
    plt.title('radial power spectrum')
    plt.xlabel("spatial scale (px)")
    plt.ylabel("power")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=200)
    plt.show()
    return
    