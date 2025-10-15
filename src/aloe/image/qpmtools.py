"""
Utility functions for Quantitative Kikuchi Pattern Matching
"""
import os
import datetime

import numpy as np
import numba
import math as m
from scipy import signal

import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.io import imread, imsave
from skimage import exposure


def img_to_uint(img, clow=0.25, chigh=99.75, dtype=np.int8):
    """ convert a numpy array to unsigned integer 8/16bit
    stretch contrast to include (clow..chigh) percent of original intensity
    """    
    p_low, p_high = np.percentile(img, (clow, chigh)) # clow, chigh in percent 0...100
    img_rescaled = exposure.rescale_intensity(img, in_range=(p_low, p_high), out_range=dtype)
    img_uint = img_rescaled.astype(dtype)    
    return img_uint 


def save_pattern(pattern_name, pattern):
    """ save numpy pattern data as txt and image
    """
    np.savetxt(pattern_name+'.txt', pattern)
    imsave(pattern_name+".png", img_to_uint(pattern, clow=1.0, chigh=99.0, dtype=np.uint8))
    imsave(pattern_name+".tif", img_to_uint(pattern, clow=1.0, chigh=99.0, dtype=np.uint16), plugin='tifffile')
    return


def load_onemask(filename):
    """ load an image and consider the zero values as mask to neglect
    """
    img = imread(filename, as_gray=True)
    # makes 0/1 mask
    onemask = img.astype(np.bool).astype(np.int) 
    return onemask


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
            
    return result


@numba.jit(nopython=True, parallel=True)
def lmsd_mean(img, kernelsize, result):
    """ local mean values in kernel
    """
    h = img.shape[0]
    w = img.shape[1]
    
    nkernelpoints = (2*kernelsize+1)**2
    for i in numba.prange(0,h):
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


@numba.jit(nopython=True, parallel=True)
def lmsd_parallel(img, kernelsize, result):
    h = img.shape[0]
    w = img.shape[1]
    
    nkernelpoints = (2*kernelsize+1)**2
    for i in numba.prange(0,h):
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
            
    return result


def lmsd_convolve(image, N):
    """ slow
    """
    im = np.array(image, dtype=float)
    im2 = np.square(im)
    ones = np.ones(im.shape)
    
    kernel = np.ones((2*N+1, 2*N+1))
    s =  signal.convolve2d(im,   kernel, mode="same", boundary='symm')
    s2 = signal.convolve2d(im2,  kernel, mode="same", boundary='symm')
    ns = signal.convolve2d(ones, kernel, mode="same", boundary='symm')

    mean = s / ns
    std = np.sqrt((s2 - s**2 / ns) / ns)
    
    image_lmsd = (im - mean) / std
    
    return image_lmsd 


def filter_lmsd(img, kernelsize_pix):
    img_temp = np.zeros_like(img)
    lmsd(img, kernelsize_pix, img_temp)
    #img_temp = lmsd_convolve(img, kernelsize_pix)
    return img_temp


def make_weight_pattern_gaussian(img, row_center=None, col_center=None, sigma_pix=None):
    """ create array of weights
    """
    weights = np.zeros_like(img, dtype=np.float32)
    height = img.shape[0]
    width = img.shape[1]
    
    if row_center is None:
        row_center = height / 2
    
    if col_center is None:
        col_center = width / 2
    
    if sigma_pix is None:
        sigma_pix = height / 4
    
    total = 0.0
    for irow in range(height):
        for icol in range(width):
            w = np.exp(-0.5 * ( (irow-row_center)**2 + (icol-col_center)**2) / sigma_pix**2 )
            weights[irow, icol] = w
            total += w
            
    # normalize
    weights[:,:] /= total
            
    return weights   

def make_weight_pattern_disk(img, row_center=None, col_center=None, r_pix=None):
    """ create array of weights
    """
    weights = np.zeros_like(img, dtype=np.float32)
    height = img.shape[0]
    width = img.shape[1]
    
    if row_center is None:
        row_center = height / 2
    
    if col_center is None:
        col_center = width / 2
    
    if r_pix is None:
        r_pix = height * 0.4
    
    for irow in range(height):
        for icol in range(width):
            r2 = (irow-row_center)**2 + (icol-col_center)**2
            if (r2 <= r_pix*r_pix):            
                weights[irow, icol] = 1.0
            else:
                weights[irow, icol] = 0.0

            
    # normalize
    weights = weights / np.sum(weights)
    
    return weights 


def pattern_signal_weighted(img, weights=None):
    """ weighted normalization of an image to have 
        weighted mean=0.0 and weighted standard deviation=1.0
        can be used directly in dot-product weighted NCC = dot(img, target)/npix
    """
    if weights is None:
        weights = np.ones_like(img)
    
    # normalize weights to total weight = 1.0
    weights = weights / np.sum(weights)
        
    height = img.shape[0]
    width = img.shape[1]
    
    # weighted mean
    mw = 0.0
    for irow in range(height):
        for icol in range(width):
            mw += weights[irow, icol] * img[irow, icol]
    #print(mw)
        
    # weighted stddev
    variance = 0.0
    for irow in range(height):
        for icol in range(width):
            variance += weights[irow, icol] * (img[irow, icol] - mw)**2
    sw = np.sqrt(variance)
    #print(sw)
    
    # shift and scale pattern signal
    signal = np.zeros_like(img)
    for irow in range(height):
        for icol in range(width):
            signal[irow, icol] = np.sqrt(weights[irow, icol]) * (img[irow, icol] - mw) / sw
     
    signal = signal * np.sqrt(img.size)
    
    return signal


def lstr(*args):
    """string representation of argument list elements as one string
    use for logging
    """
    list_string = ' '.join(map(str, args)) 
    return list_string


def time_id():
    """ current time as string 
    e.g. use for output directory and filenames
    """
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    second = '{:02d}'.format(now.second)
    day_month_year_time = '{}{}{}_{}{}{}'.format(year, month, day,hour,minute,second)
    return day_month_year_time
    
    
def checkmakedir(folder):
    """ creates a directory if it does not exist
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    return
    
    
def split_ext(filename):
    """ return path without extension and extension
    """
    ext = os.path.splitext(filename)[1]
    path_no_ext = os.path.splitext(filename)[0]
    return path_no_ext, ext    


def print_statistics(array, title=None):
    if title is not None:
        print(title)
    print(np.max(array), np.mean(array), np.min(array))
    return


def load_target_pattern(pattern_filename, weights=None, mask_filename=None, r_mask=1.5, lmsd=None):   
    """ load and filter pattern
    """
    img_pattern = np.array(imread(pattern_filename, as_gray=True), dtype=float)
    
    if lmsd is not None:
        kernelsize_pix = int(lmsd * img_pattern.shape[1])
        img_pattern = filter_lmsd(img_pattern, kernelsize_pix)
    
    if weights is None:
        weights = make_weight_pattern_disk(img_pattern, r_pix = r_mask * img_pattern.shape[1])
        #print_statistics(weights, "weights")
        
    #LOG.info( lstr("Loaded Pattern Width, Height: ", img_pattern.shape) )
    #h, w = img_pattern.shape
    
    #img_pattern = process_pattern_tsl(img_pattern, sigma=0.1, rmax=r_mask)
    
    #print(np.max(img_pattern), np.min(img_pattern))
    
    #img_target = make_normalized_masked_image(img_pattern, rmax=r_mask)
    #print(img_pattern.dtype)
    #img_exp = make_normalized_masked_image(img_pattern, rmax=r_mask)
    #plot_image(img_exp, limits=limits,
    #            title=str(1.0), cmap="inferno",
    #            filename="current_target.png",show=False)
    #img_exp = filter_img(img_pattern)
    img_target = pattern_signal_weighted(img_pattern, weights)
    return img_target


def ncc_weighted(img1, img2, weights, 
        plot_filename=None,
        plot_limits=[1.9, 1.9],
        savetxt=False):

    signal1 = pattern_signal_weighted(img1, weights)
    signal2 = pattern_signal_weighted(img2, weights)

    signal1 = np.nan_to_num(signal1, nan=0.0)
    signal2 = np.nan_to_num(signal2, nan=0.0)

    xc = np.dot(np.ravel(signal1), np.ravel(signal2)) / signal1.size

    if plot_filename is not None:

        plot_img_diff(signal1, signal2, 
            imgrange=plot_limits[0],
            asyrange=plot_limits[1],
            filename=plot_filename, 
            title="NCC:  %.4f" % xc )
        
    if savetxt:
        asy =(signal1 - signal2)
        np.savetxt("current_exp.txt", exp)
        np.savetxt("current_sim.txt", sim)
        np.savetxt("current_asy.txt", asy)
        
    return xc


def plot_img_diff(exp, sim, filename=None, 
        imgrange=2.9, asyrange=2.9,
        cmap_pattern='inferno', cmap_diff='bwr',
        xlim=None, ylim=None, title=None,
        horizontal = False, dpi=200, show=False):
    """ plot experiment, simulation and make difference of normalized patterns"""

    def format_colorbar(cb, label=None, fontsize=8):
        """ apply tick label formating
            cb = fig.colorbar()
        """
        
        def format_func(value, pos):
            """ format the color bar labels """
            if (value>0):
                # plus sign
                x = "+" + '{:0.0f}'.format(abs(value))
            elif (value<0):
                # wide minus sign
                x = u"\u2212" + '{:0.0f}'.format(abs(value))
            else:
                # zero
                x = '{:0.0f}'.format(abs(value)) # label string
            
            if x.startswith('-') or x.startswith('+'):       
                pad = ''
            else: 
                pad = ' '
            return '{}{}'.format(pad, x)
            
        cb.ax.tick_params(labelsize=fontsize)
        if label: cb.set_label(label,size=fontsize, fontweight='normal', labelpad=4)
        cb.ax.yaxis.major.formatter = matplotlib.ticker.FuncFormatter(format_func)
        for tick in cb.ax.get_yticklabels():
            # tick is Text
            #tick.set_rotation(90)
            tick.set_horizontalalignment("right")
            tick.set_verticalalignment("center")
        cb.ax.yaxis.set_tick_params(pad=13)  # your number may vary
        return


    


    #def set_label_size(ax, fontsize):
    #    """ set font size of axis labels
    #    """
    #    ax.xaxis.label.set_size(fontsize)
    #    ax.yaxis.label.set_size(fontsize)



    h, w = exp.shape

    labelsize = 8

    #exp = standardize_image_onemask(exp, onemask=onemask) 
    #sim = standardize_image_onemask(sim, onemask=onemask)

    #exp = exp * np.sqrt(npix)
    #sim = sim * np.sqrt(npix)

    if horizontal:
        fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, 
            figsize=(5.0 * h/w *1.05, 3), dpi=200, facecolor='w', constrained_layout=False)
        #plt.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.95, hspace=0.05,
        #            wspace=0.25)
    else:
        fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, 
            figsize=(3, 5.0 * h/w *1.05), dpi=200, facecolor='w',constrained_layout=False)
    
        #plt.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.95, hspace=0.05,
        #            wspace=0.05)
    ax1, ax2, ax3 = axes
    
    # no tick lines sticking out
    for ax in axes:
        ax.tick_params(axis='both', which='both', length=0)
    
    
    exp_plt= ax1.imshow(exp, zorder=-10, vmin=-imgrange, vmax=imgrange, cmap=cmap_pattern)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    #colorbar_format = '% 1.0f'
    cbar1 = fig.colorbar(exp_plt, cax=cax1, orientation='vertical', pad=0.05) #, format=colorbar_format)
    format_colorbar(cbar1, label='A')

    sim_plt=ax2.imshow(sim, zorder=-8,alpha=1.0, vmin=-imgrange, vmax=imgrange, cmap=cmap_pattern)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(sim_plt, cax=cax2, orientation='vertical', pad=0.05)
    format_colorbar(cbar2, label='B')
    
    asy=(exp - sim)
    asy_plt=ax3.imshow(asy, zorder=-6, cmap=cmap_diff, vmin=-asyrange, vmax=asyrange)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig.colorbar(asy_plt, cax=cax3, orientation='vertical', pad=0.1)
    format_colorbar(cbar3, label='A - B')
    
    
    for ax in axes:
        #set_label_size(ax, label_size)
        ax.axis('equal')
        ax.set_aspect('equal', 'datalim')
        if xlim is None:
            ax.set_xlim(0,w)
        else:
            ax.set_xlim(xlim)
        if ylim is None:
            ax.set_ylim(h, 0)
        else:
            ax.set_ylim(ylim)
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    if title:
        plt.suptitle(title, fontsize=labelsize) #, va="bottom", ha="center", y=0.96, x=0.51)
    #plt.ylim(exp.shape[0],0)
    #plt.xlim(0,exp.shape[1])
    #plt.tight_layout(h_pad=20)
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        
    if show:
        plt.show()
        
    plt.close()
    return


def plot_image(image=None, limits=None, title=None, 
    cmap="Greys_r", filename=None, labels=True, show=True,
    xlim=None, ylim=None, extent=None, tick_spacing=None,
    xlabel=None, ylabel=None, clabel=None):
    """
    simple image plot with color bar, optional title and limits on values
    """
    if image is not None:
        fig, ax = plt.subplots()
        fig.patch.set_alpha(1)
        
        if not labels:
            ax.tick_params(labelbottom=False, labelleft=False, direction="in")
        divider = make_axes_locatable(ax)
        
        cax = divider.append_axes('right', size='5%', pad=0.05)
        
        if limits is None:
            im=ax.imshow(image, cmap=cmap, extent=extent)
        else:
            im=ax.imshow(image, vmin=limits[0], vmax=limits[1], 
                cmap=cmap, extent=extent)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)
        
        if tick_spacing is not None:
            ax.xaxis.set_major_locator(mtick.MultipleLocator(tick_spacing))
            ax.yaxis.set_major_locator(mtick.MultipleLocator(tick_spacing))

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=14)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=14)

        if title is not None:    
            ax.set_title(title)

        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        if (clabel is not None):
            cb.set_label(clabel, fontsize=12, labelpad=10)
            

        if (filename is not None):
            try:
                plt.savefig(filename, dpi=300, bbox_inches = 'tight', transparent=False, facecolor='white')
            except OSError:
                print("WARNING: Could not save file: ", filename)
                
        if show:
            plt.show()
        else:
            plt.close()
    return


def pcwwiu2pcgnom(wwiu_PCX, wwiu_PCY, wwiu_PCZ, pattern_height, pattern_width):
    """ limits of gnomonic projection in WWIU convention
    """
    screen_aspect=float(pattern_width)/float(pattern_height)
    x_gn_min = -((    wwiu_PCX)*screen_aspect) / wwiu_PCZ    
    x_gn_max = +((1.0-wwiu_PCX)*screen_aspect) / wwiu_PCZ
    y_gn_min = - (1.0-wwiu_PCY)                / wwiu_PCZ
    y_gn_max = + (    wwiu_PCY)                / wwiu_PCZ
    return [x_gn_min, x_gn_max, y_gn_min, y_gn_max]

    
def pcoina2pcgnom(oina_PCX, oina_PCY, oina_DD, pattern_height, pattern_width):
    """ limits of gnomonic projection in OINA convention
    """
    #screen_aspect=float(pattern_width)/float(pattern_height)
    x_gn_min = -(    oina_PCX) / oina_DD    
    x_gn_max = +(1.0-oina_PCX) / oina_DD
    y_gn_min = -(    oina_PCY) / oina_DD
    y_gn_max = +(pattern_height - oina_PCY * pattern_width) / (oina_DD * pattern_width)
    return [x_gn_min, x_gn_max, y_gn_min, y_gn_max]


def filter(I, K, cache=None):
    """
    Filter image I with kernel K.

    Image color values outside I are set equal to the nearest border color on I.

    To filter many images of the same size with the same kernel more efficiently, use:

      >>> cache = []
      >>> filter(I1, K, cache)
      >>> filter(I2, K, cache)
      ...

    An even width filter is aligned by centering the filter window around each given
    output pixel and then rounding down the window extents in the x and y directions.
    """
    def roundup_pow2(x):
      y = 1
      while y < x:
        y *= 2
      return y

    I = np.asarray(I)
    K = np.asarray(K)

    if len(I.shape) == 3:
      s = I.shape[0:2]
      L = []
      ans = np.concatenate([filter(I[:,:,i], K, L).reshape(s+(1,))
                               for i in range(I.shape[2])], 2)
      return ans
    if len(K.shape) != 2:
      raise ValueError('kernel is not a 2D array')
    if len(I.shape) != 2:
      raise ValueError('image is not a 2D or 3D array')

    s = (roundup_pow2(K.shape[0] + I.shape[0] - 1),
         roundup_pow2(K.shape[1] + I.shape[1] - 1))
    Ipad = np.zeros(s)
    Ipad[0:I.shape[0], 0:I.shape[1]] = I

    if cache is not None and len(cache) != 0:
      (Kpad,) = cache
    else:
      Kpad = np.zeros(s)
      Kpad[0:K.shape[0], 0:K.shape[1]] = np.flipud(np.fliplr(K))
      Kpad = np.fft.rfft2(Kpad)
      if cache is not None:
        cache[:] = [Kpad]

    Ipad[I.shape[0]:I.shape[0]+(K.shape[0]-1)//2,:I.shape[1]] = I[I.shape[0]-1,:]
    Ipad[:I.shape[0],I.shape[1]:I.shape[1]+(K.shape[1]-1)//2] = I[:,I.shape[1]-1].reshape((I.shape[0],1))

    xoff = K.shape[0]-(K.shape[0]-1)//2-1
    yoff = K.shape[1]-(K.shape[1]-1)//2-1
    Ipad[Ipad.shape[0]-xoff:,:I.shape[1]] = I[0,:]
    Ipad[:I.shape[0],Ipad.shape[1]-yoff:] = I[:,0].reshape((I.shape[0],1))

    Ipad[I.shape[0]:I.shape[0]+(K.shape[0]-1)//2,I.shape[1]:I.shape[1]+(K.shape[1]-1)//2] = I[-1,-1]
    Ipad[Ipad.shape[0]-xoff:,I.shape[1]:I.shape[1]+(K.shape[1]-1)//2] = I[0,-1]
    Ipad[I.shape[0]:I.shape[0]+(K.shape[0]-1)//2,Ipad.shape[1]-yoff:] = I[-1,0]
    Ipad[Ipad.shape[0]-xoff:,Ipad.shape[1]-yoff:] = I[0,0]

    ans = np.fft.irfft2(np.fft.rfft2(Ipad) * Kpad, Ipad.shape)

    off = ((K.shape[0]-1)//2, (K.shape[1]-1)//2)
    ans = ans[off[0]:off[0]+I.shape[0],off[1]:off[1]+I.shape[1]]

    return ans


def gaussian(sigma=0.5, shape=None):
    """
    Gaussian kernel numpy array with given sigma and shape.
    
    The shape argument defaults to ceil(6*sigma).
    """
    sigma = max(abs(sigma), 1e-10)
    if shape is None:
      shape = max(int(6*sigma+0.5), 1)
    if not isinstance(shape, tuple):
      shape = (shape, shape)
    x = np.arange(-(shape[0]-1)/2.0, (shape[0]-1)/2.0+1e-8)
    y = np.arange(-(shape[1]-1)/2.0, (shape[1]-1)/2.0+1e-8)
    Kx = np.exp(-x**2/(2*sigma**2))
    Ky = np.exp(-y**2/(2*sigma**2))
    ans = np.outer(Kx, Ky) / (2.0*np.pi*sigma**2)
    return ans/sum(sum(ans))    
    
    
def blur_gaussian(image, sigma=None, support=None):
    """ 
    Gaussian blur of image
    
    sigma   : sigma of Gaussian
    support : extension of Kernel, should be a few sigmas
    """
    if sigma is None:
        sigma=image.shape[1]/8
    
    if support is None:
        support=(3*sigma, 3*sigma)
    
    img_blurred = filter(image, gaussian(sigma,support))
    return img_blurred
    
    
def test_plot():
    print("Making Test Plot...")
    img_dim = (300, 400)
    img1 = np.random.normal(size=img_dim)
    img2 = np.random.normal(size=img_dim)
    onemask = np.ones_like(img1)
    print(img1.shape, img2.shape, onemask.shape)
    filename = "testplot.png"
    xc_onemask(img1, img2, onemask, plot_filename=filename)
    return
    
if __name__ == "__main__":
    test_plot()
    
        
    
    
    