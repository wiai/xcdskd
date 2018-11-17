import numpy as np
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.util.dtype import dtype_range
from skimage.util import random_noise
from scipy.ndimage.filters import gaussian_filter

from ..image.utils import img_to_uint
from ..image.nxcc import standard_img

def makeGaussian(nwidth, nheight, fwhm = None, center=None, yfac=1.0):
    """ Gaussian 2D image
    fwhm is full-width-half-maximum
    yfac stretches hoizontal relative to vertical
    peak value will be 1.0
    """
    # matrix of the x values   
    x= np.arange(0, nwidth)[np.newaxis,:]*np.ones((nheight,nwidth), dtype=np.float) 
    # matrix of the y values   
    y= np.arange(0,nheight)[:,np.newaxis]*np.ones((nheight,nwidth), dtype=np.float)    
    
    #print(x)
    #print(y)
    
    if fwhm is None:
        fwhm = nwidth  
   
    if center is None:
        x0 = nwidth // 2 
        y0 = nheight // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (yfac*(y-y0))**2) / fwhm**2)

def norm_to_max(img):
    return img/np.max(img)

def norm_to_range(img):
    img_min=np.min(img)
    img_max=np.max(img)
    return (img-img_min)/(img_max-img_min)

def normalized_ubyte_from_data(data):
    img_ubyte=img_as_ubyte(data)
    return img_ubyte

    
def ebsp_detection_model_poisson(bkd, bg_fwhm=0.8, bg_offset=0.01, 
                      bkd_signal=0.1, 
                      counts=20000):
    """
    counting noise simulation for EBSP
    model for experimental ebsp intensity with blur & noise
    """
    eps=0.00001
    
    # normalize bkd signal 0...1
    bkd_norm=norm_to_range(bkd) 
    bkd_dev=np.std(bkd_norm)
    # print(bkd_dev)
    
    # background intensity= 2D gaussian 0..1
    # physics: incoherent inelastic scattering cross section
    bg = norm_to_max( makeGaussian(bkd.shape[1], bkd.shape[0], fwhm=bkd.shape[0]*bg_fwhm)
                      + bg_offset )
    
    # analog signal: BG + proportional BKD
    # physics: assume coherent/incoherent is not dependent on angle in BKD
    ebsp_ideal_norm = norm_to_max(  (1.0-bkd_signal)*bg + bg * bkd_signal * bkd_norm)
    
    # counting signal = counts with Poisson distribution according to mean number of collected counts
    
    ebsp_poisson_counts = np.random.poisson(lam=counts*ebsp_ideal_norm)  

    return ebsp_poisson_counts,  ebsp_ideal_norm


def ed_pattern(pattern, mix=1.0,  gradient_blur=3.0, gradient_clip=2.0, dtype=np.uint8):
    """ empirical excess-deficiency effect
    """
    pattern_gradient = np.gradient(pattern)
    # blur and amplify gradient
    ed_add = np.clip(standard_img(gaussian_filter(pattern_gradient[0], gradient_blur)),
                     -gradient_clip, gradient_clip)
    # add variation from gradient
    ed_pattern = img_to_uint(standard_img(pattern) + mix*ed_add, dtype=dtype)
    return ed_pattern

    
def ebsp_detection_model(bkd, static_bg = None, bg_fwhm=0.8, bg_offset=0.01, 
                      theory_blur=2.0,
                      bkd_signal=0.1, 
                      ed_mix=1.0,  gradient_blur=3.0, gradient_clip=2.0,
                      counts=20000, 
                      gauss_std=0.1, gauss_mean=0.01,
                      
                      lens_blur=0.001,
                      cam_offset=0.0, cam_gain=1.0, cam_ngray=256, cam_gnoise=1.0,
                      dtype=np.uint8, mirror_ud=False):
    """
    BKD -> simulated experimental EBSP
    model for experimental ebsp intensity with blur & noise
    """
    eps=0.00001
    
    # blur the theoretical pattern
    bkd = gaussian_filter(bkd, theory_blur) 
    
    # E-D effekt on pure theory
    bkd = ed_pattern(bkd, mix=ed_mix, gradient_blur=gradient_blur, gradient_clip=gradient_clip, dtype=dtype)
    
    # normalize bkd signal 0...1
    bkd_norm=norm_to_range(bkd) 
    bkd_dev=np.std(bkd_norm)
    # print(bkd_dev)
    
    # background intensity= 2D gaussian 0..1
    # physics: incoherent inelastic scattering cross section
    bg = norm_to_max( makeGaussian(bkd.shape[1], bkd.shape[0], fwhm=bkd.shape[0]*bg_fwhm)
                      + bg_offset )
    
    # analog signal: BG + proportional BKD
    # physics: assume coherent/incoherent is not dependent on angle in BKD
    ebsp_ideal_norm = norm_to_max(  (1.0-bkd_signal)*bg + bg * bkd_signal * bkd_norm)
    
    if static_bg is not None:
        ebsp_ideal_norm *= norm_to_max(static_bg)
    
    # counting signal = counts with Poisson distribution according to mean number of collected counts
    
    ebsp_poisson = np.random.poisson(lam=counts*ebsp_ideal_norm)   
   
    # renormalize to mean_counts
    ebsp_poisson = ebsp_poisson / counts
    
    # additive Gaussian noise (phosphor) with offset gmean and stddev gnoise
    # offset by mean
    if (gauss_std>eps):
        gaussian_noise = np.random.normal(loc=gauss_mean, scale=np.abs(gauss_std*np.ones_like(bkd)))
    else:
        gaussian_noise = np.zeros_like(bkd)    
    
    # optical blur
    phosphor = ebsp_poisson + gaussian_noise
    phosphor_optics = gaussian_filter(phosphor, lens_blur)  
                                          
    # camera gaussian noise: (thermal, electronics)
    if (cam_gnoise>eps):
        cam_gauss = np.random.normal(loc=cam_gnoise, scale=cam_gnoise*np.ones_like(bkd))  
    else:
        cam_gauss = np.zeros_like(bkd)    
    
      
                                          
    # gain, offset, clip signal to digitizer range with gray levels ngray (256, or e.g. 16bit)
    ebsp_int = np.rint( cam_ngray * (cam_gain*phosphor_optics + cam_offset + cam_gauss))
    
    ebsp_clip = np.clip(ebsp_int, 0, cam_ngray-1)
    
    #ebsp_int = bkd_norm
    if mirror_ud:
        # mirror final image up down e.g. for optimus detection
        ebsp_clip=np.flipud(ebsp_clip)
    
    #ebsp_uint = img_to_uint(ebsp_clip, clow=0.01, chigh=99.99, dtype=dtype)
        
    return ebsp_clip

    
    
def ebsp_detection_model1(img):
    return ebsp_detection_model(img, bg_fwhm=0.8, bg_offset=0.1, 
                      bkd_signal=0.3, 
                      counts=10000, 
                      gauss_std=0.01, gauss_mean=0.01,
                      lens_blur=0.5,
                      cam_offset=0.0, cam_gain=1.0, cam_ngray=256)

