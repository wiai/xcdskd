"""
normalized cross correlation coefficient calculation routines
"""

import numpy as np
import numpy.ma as ma
import numba
import math 
from PIL import Image

# nopython=True means an error will be raised
# if fast compilation is not possible.
@numba.jit(nopython=True)  
def norm_img(img):
    """normalize image vector to have mean=0 and stddev=1/np.sqrt(img.size)
    """
    mean_img = np.mean(img)
    std_img = np.std(img)
    return (img-mean_img)/(std_img*np.sqrt(img.size))

    
@numba.jit(nopython=True)  
def calc_xc(img1,img2):
    """ normalized cross correlation coefficient
    """
    return np.sum(norm_img(img1)*norm_img(img2)) 
    
    
@numba.jit(nopython=True)  
def standard_img(img):
    """normalize image vector to have mean=0 and stddev=1
    """
    mean_img = np.mean(img)
    std_img = np.std(img)
    return (img-mean_img)/std_img
    

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
    #print(mimg.size, npixels, mean_mimg, std_mimg )
    mimg = (mimg - mean_mimg)/(std_mimg*np.sqrt(npixels))
    #print('normalized pattern: ', mimg.size, npixels, np.mean(mimg), np.std(mimg)*np.sqrt(npixels) )
    return mimg, npixels

def standard_img_mask(img, mask=None):
    """ normalize image to have mean=0 and stddev=1
        take only positive definite pixel values
    """
    stdimage, npixels = norm_img_mask(img, mask=mask)
    stdimage *= np.sqrt(npixels) 
    return stdimage

@numba.jit(nopython=True)    
def ntxc(NormRefImg, TestImg):
    """ normalized cross-correlation coefficient
    

    NormRefImg : normalized reference image (e.g. experimental pattern),
        avoids repeated normalization of NormRefImg
        
    TestImg: trial image to be tested for XC, will be normalized
    
    """
    return np.sum(NormRefImg*norm_img(TestImg))
    

    
@numba.jit((numba.float64(numba.float64[:,:], numba.float64[:,:])), nopython=True, cache=True)
def dotprod(arr1, arr2):
    """ sum of element-wise product of two arrays
    
    Parameters:

    arr1, arr2 :   2D  float arrays
    """
    M, N = arr1.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr1[i,j] * arr2[i,j]
    return result



@numba.jit(nopython=True)    
def nnxc(NormRefImg, NormTestImg):
    """ cross-correlation coefficient between two NORMALIZED (mean=0, stddev=1) image matrices
    
    Parameters:

    NormRefImg :    normalized reference image (e.g. experimental pattern),
                    avoids repeated normalization of NormRefImg 
    NormTestImg:    normalized   trial image to be tested for XC
                    make sure that both have same dimensions
    """
    return np.sum(NormRefImg*NormTestImg)


@numba.jit(nopython=True)     
def asy(im1, im2):
    """Asymmetry image: difference normalized by sum
    """
    return (im1-im2)/(im1+im2)


@numba.jit(nopython=True)
def mask_pattern_disk(pattern, rmax=0.5, rmin=-0.5, cx_outer=0.5, cy_outer=0.5, cx_inner=0.5, cy_inner=0.5):
    """ apply circular masks to pattern
    radius in terms of half pattern width
    
    the mask is given by values which are 0
    
    """
    masked_pattern=np.copy(pattern)
    width=pattern.shape[1]
    height=pattern.shape[0]

    cx_outer = cx_outer * width    
    cy_outer = cy_outer * height      
    cx_inner = cx_inner * width    
    cy_inner = cy_inner * height   
    rmax = rmax * width
    rmin = rmin * width
    npixels=0
    for iy in range(height):
        for ix in range(width):
            rxy_outer = math.sqrt((ix-cx_outer)**2 + (iy-cy_outer)**2)
            rxy_inner = math.sqrt((ix-cx_inner)**2 + (iy-cy_inner)**2)
            if (rxy_outer > rmax) or (rxy_inner < rmin):
                masked_pattern[iy, ix] = 0.0
            else:
                npixels += 1
    return masked_pattern, npixels


def count_pixels(image):
    """
    count pixels with values larger than zero
    """  
    npixels = np.sum(image>0.0)
    return npixels


def load_image(pattern_filename):
    """ load image from file as greyscale floats """
    return np.array(Image.open(pattern_filename).convert('L')).astype(np.float32)

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    
    Arguments:

        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
            
    Returns:

        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)