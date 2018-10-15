import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable

#import pylab

import numpy as np



from .image import arbse 
from .image.arbse import normalizeChannel, make2Dmap, get_vrange    
from .image.nxcc import norm_img, norm_img_mask, nnxc, mask_pattern_disk, count_pixels    
    
    
def get_vrange_pos(signal, stretch=2.0):
    med=np.median(signal)
    std=np.std(signal)
    vmin=np.max([med-stretch*std, 0.0])
    vmax=med+stretch*std
    return [vmin,vmax]   
    
    
def plot_image(image=None, limits=None, title=None, 
    cmap="Greys_r", filename=None, labels=True):
    """
    simple image plot with color bar, optional title and limits on values
    """
    if image is not None:
        fig, ax = plt.subplots()
        if not labels:
            ax.tick_params(labelbottom=False, labelleft=False, direction="in")
        divider = make_axes_locatable(ax)
        
        cax = divider.append_axes('right', size='5%', pad=0.05)
        
        if limits is None:
            im=ax.imshow(image, cmap=cmap)
        else:
            im=ax.imshow(image, vmin=limits[0], vmax=limits[1], cmap=cmap)
        if title is not None:    
            ax.set_title(title)
        fig.colorbar(im, cax=cax, orientation='vertical')
        if (filename is not None):
            plt.savefig(filename, dpi=300, bbox_inches = 'tight')
        plt.show()
    return
    
    
def plot_pattern(data, vrange=None, filename=None):
    plot_SEM(data, ticklabels=False, scalebar=False, vrange=vrange, filename=filename)
    
    
    
def plot_SEM(data, vrange=None, title=None, colorbar=True, 
             cmap='viridis', colorbarlabel=None, filename=None,
             cootype=None, rot180=False, microns=1.0, 
             xlabel=None, ylabel=None, ticklabels=True, scalebar=True):
    
    from matplotlib import rcParams
    from matplotlib_scalebar.scalebar import ScaleBar
    saved_rcparams = (rcParams['figure.figsize'], rcParams['font.size'], 
        rcParams['xtick.direction'], rcParams['ytick.direction'])

    
    rcParams['figure.figsize'] = (8.0, 5.0)
    rcParams['font.size'] = (12)
    
    # Matplotlib option: set ticks outside of axis in plot
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    
    if vrange is None:
        vrange=get_vrange(data)
    MinValue, MaxValue = vrange  

    
    width = microns * data.shape[1]
    height= microns * data.shape[0]
    extent=[0, width, height, 0] # from lower left
    
    plt.figure()
    ax = plt.subplot(111)
    ax.set_aspect('equal')    
    
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=22)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=22)
    
    #if cootype is None:
    #    plotdata=np.flipud(data) # because of extent
    plotdata=data

    #if rot180:
    #    origin='lower', 'upper'
    if cootype=='XYSAMPLE':
        plotdata=np.flipud(np.fliplr(data))
        #xlabel=r'$\mathrm{X_{S}}$ $\mathrm{(\mu m)}$',
        #ylabel=r'$\mathrm{Y_{S}}$ $\mathrm{(\mu m)}$'
        
    #plt.title(TitleString, y=1.02)
        
    if MinValue==0 and MaxValue==0:
        plt.imshow(plotdata, interpolation='nearest', cmap=cmap) #, extent=extent)  
    else:
        plt.imshow(plotdata, interpolation='nearest', cmap=cmap,
            vmin=MinValue, vmax=MaxValue) #, extent=extent)
    
    
    if scalebar:
        scale = ScaleBar(dx=microns, units="um", location="lower left", 
            border_pad=0.4, pad=0.3, box_color='k', color='w',
            box_alpha=0.5, length_fraction=0.24) # 1 pixel = dx units
        ax.add_artist(scale)
    
    #ax.invert_yaxis()

    #if ((cootype=='XYSAMPLE') and (rot180==False)) or ((cootype=='BEAMIX')  and (rot180==True)):
    #    plt.gca().invert_xaxis()
    #    plt.gca().invert_yaxis()
    
    

    if colorbar:
        #set color bar to same height as plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb=plt.colorbar(cax=cax)
        #
        if (colorbarlabel is not None):
            cb.set_label(colorbarlabel, fontsize=18, labelpad=10)
        #
        
        #ticklabs = cb.ax.get_yticklabels()
        ###cb.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%+6.2f'))  #'%+06.2f'      
        #cb.ax.set_yticklabels(ticklabs, ha='right')
        #cb.ax.yaxis.set_tick_params(pad=45)   

    if not ticklabels:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    plt.draw()
    if (filename is not None):
        plt.savefig(filename+'.png', dpi=300, bbox_inches = 'tight')
     
    plt.show()
    
    # reset figure size to what it was
    rcParams['figure.figsize'], rcParams['font.size'], rcParams['xtick.direction'], rcParams['ytick.direction'] = saved_rcparams
    return
    


def plot_SEM_RGB(red, blue, green, maprows, mapcols, 
        rot180=False, add_bright=0, contrast=1.0, filename=None,
        microns=1.0, ticklabels=False):
    """
    colors are the normalized intensities on each FSD
    """

    rgb = np.zeros((maprows,mapcols,3), 'uint8')
    
    rgb[..., 0] = normalizeChannel(red,   AddBright=add_bright, Contrast=contrast)
    rgb[..., 1] = normalizeChannel(blue,  AddBright=add_bright, Contrast=contrast)
    rgb[..., 2] = normalizeChannel(green, AddBright=add_bright, Contrast=contrast)
    #if rot180:
        #rgbArray=np.rot90(rgbArray)
        #rgbArray=np.rot90(rgbArray)
    plot_SEM(rgb, vrange=None, colorbar=None,
        colorbarlabel=None, filename=filename, rot180=rot180,
        ticklabels=ticklabels, microns=microns)

    return rgb
    
    
# ARBSE 7x7

def plot_arbse_rows(vFSD, signal_name, XIndex, YIndex, MapHeight, MapWidth):
    """ signal: sum of row in 7x7 array
    """
    vmin=1e12
    vmax=0
    bse_rows = []
    # (1) get full range for all images
    for row in range(7):
        signal = np.sum(vFSD[:,row,:], axis=1) #/vFSD[:,row+drow,0]
        signal_map = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)
        minv, maxv = get_vrange(signal)
        if (minv<vmin):
            vmin=minv
        if (maxv>vmax):
            vmax=maxv    

    # (2) make plots with same range for comparisons of absolute BSE values
    vrange=[vmin, vmax]
    #print(vrange)
    for row in range(7):
        signal = np.sum(vFSD[:,row,:], axis=1) #/vFSD[:,row+drow,0]
        signal_map = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)
        bse_rows.append(signal_map)
        plot_SEM(signal_map, vrange=vrange, filename=signal_name+'_row_'+str(row), rot180=True)
    return bse_rows
        
        
def plot_arbse_columns(vFSD, signal_name, XIndex, YIndex, MapHeight, MapWidth):
    """
    signal: sum of column in 7x7 vFSD array
    """
    vmin=1e12
    vmax=0
    bse_cols = []

    # (1) get full range for all images
    for col in range(7):
        signal = np.sum(vFSD[:,:,col], axis=1) #/vFSD[:,row+drow,0]
        signal_map = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)
        minv, maxv = get_vrange(signal)
        if (minv<vmin):
            vmin=minv
        if (maxv>vmax):
            vmax=maxv    

    # (2) make plots with same range for comparisons of absolute BSE values
    #vrange=[vmin, vmax]
    vrange=None # no fixed scale

    for col in range(7):
        signal = np.sum(vFSD[:,:,col], axis=1) #/vFSD[:,row+drow,0]
        signal_map = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)
        bse_cols.append(signal_map)
        plot_SEM(signal_map, vrange=vrange, filename=signal_name+'_col_'+str(col), rot180=True)
    return bse_cols   
        
        
def plot_arbse_rgb(vFSD, signal_name, XIndex, YIndex, MapHeight, MapWidth):
    """ rgb direct signals from left, middle, right of 7x7 array
    """
    rgb_direct = []
    for row in range(7):
        signal = vFSD[:,row,0]
        red = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)

        signal = vFSD[:,row,3]
        green = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)

        signal = vFSD[:,row,6]
        blue = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)

        rgb=plot_SEM_RGB(red, green, blue, MapHeight, MapWidth, 
                         filename=signal_name+'_RGB_row_'+str(row),
                         rot180=False, add_bright=0, contrast=0.8)

        rgb_direct.append(rgb)
    return rgb_direct

    
def plot_arbse_row_ratio(vFSD, signal_name, XIndex, YIndex, MapHeight, MapWidth, microns=1.0):
    """ relative change to previous row in 7x7 array, start at row 1
    """
    rgb_row_ratio=[]
    for row in range(1, 7):
        drow = -1
        signal = vFSD[:,row,0]/vFSD[:,row+drow,0]
        red = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)

        signal = vFSD[:,row,3]/vFSD[:,row+drow,3]
        green = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)

        signal = vFSD[:,row,6]/vFSD[:,row+drow,6]
        blue = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)

        rgb=plot_SEM_RGB(red, green, blue, MapHeight, MapWidth, 
                         filename=signal_name+'_RGB_row_ratio_'+str(row),
                         rot180=False, add_bright=0, contrast=1.2, 
                         ticklabels=False, microns=microns)      
        rgb_row_ratio.append(rgb)
    return rgb_row_ratio
                         


def plot_arbse_row_asy(vFSD, signal_name, XIndex, YIndex, MapHeight, MapWidth):
    """ signal:asymmetry to previous row 
    """
    rgb_row_asy=[]
    for row in range(1,7):
        drow=-1
        signal = arbse.asy(np.sum(vFSD[:,row,:], axis=1) , np.sum(vFSD[:,row+drow,:], axis=1))
        signal_map = make2Dmap(signal,XIndex,YIndex,MapHeight,MapWidth)
        rgb = plot_SEM(signal_map, vrange=None, cmap='gray', 
                 colorbarlabel='asymmetry',
                 filename=signal_name+'_row_asy_'+str(row))   
        rgb_row_asy.append(rgb)
    return rgb_row_asy 
    
    
    
def plot_pattern_norm_diff(exp, sim, filename=None, 
        imgrange=2.8, diff_range=9.0,
        xlim = None, ylim=None, layout='row'):
    """ plot experiment, simulation and make difference of normalized patterns"""

    h, w = exp.shape

    label_size = 4
    plt.rcParams['xtick.labelsize'] = label_size 
    plt.rcParams['ytick.labelsize'] = label_size
    exp, npix = norm_img_mask(exp)
    sim, npix = norm_img_mask(sim, mask=exp.mask)

    exp = exp * np.sqrt(npix)
    sim = sim * np.sqrt(npix)

    #fig= plt.figure(figsize=(4, 8), dpi=150)
    if layout=='col':
        # 1 column layout
        fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, 
            figsize=(3, 5.0), dpi=150, facecolor='w')
        plt.subplots_adjust(top=0.98, bottom=0.06, left=0.08, right=0.92, hspace=0.05,
                    wspace=0.05)
    if layout=='row':
        fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, 
            figsize=(5.0, 3.0), dpi=300, facecolor='w')
        plt.subplots_adjust(top=0.98, bottom=0.06, left=0.08, right=0.92, hspace=0.1,
                    wspace=0.3)

    ax1, ax2, ax3 = axes
    exp_plt= ax1.imshow(exp, zorder=-10, vmin=-imgrange, vmax=imgrange)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(exp_plt, cax=cax1, orientation='vertical')


    sim_plt=ax2.imshow(sim, alpha=1.0, vmin=-imgrange, vmax=imgrange)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sim_plt, cax=cax2, orientation='vertical')

    asy=(exp - sim)
    asy_plt=ax3.imshow(asy, cmap="seismic", vmin=-diff_range, vmax=diff_range)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(asy_plt, cax=cax3, orientation='vertical')

    for ax in axes:
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

    #plt.ylim(exp.shape[0],0)
    #plt.xlim(0,exp.shape[1])
    #plt.tight_layout(h_pad=1)
    plt.draw()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    return
    
    
    
def plot_pattern_ratio(exp, sim, filename=None,
        pattern_range=None, ratio_range= None,
        xlim = None, ylim=None, layout='row'):
    """ plot experiment, simulation and ratio of patterns"""

    h, w = exp.shape

    label_size = 4
    plt.rcParams['xtick.labelsize'] = label_size 
    plt.rcParams['ytick.labelsize'] = label_size
    #exp, npix = norm_img_mask(exp)
    #sim, npix = norm_img_mask(sim, mask=exp.mask)

    #exp = np.ma.masked_less_equal(exp, 0)
    #sim = np.ma.masked_less_equal(sim, 0)

    #fig= plt.figure(figsize=(4, 8), dpi=150)
    if layout=='col':
        # 1 column layout
        fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, 
            figsize=(3, 5.0), dpi=300, facecolor='w')
        plt.subplots_adjust(top=0.98, bottom=0.06, left=0.08, right=0.92, hspace=0.05,
                    wspace=0.05)
    if layout=='row':
        fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, 
            figsize=(5.0, 3.0), dpi=300, facecolor='w')
        plt.subplots_adjust(top=0.98, bottom=0.06, left=0.08, right=0.92, hspace=0.1,
                    wspace=0.3)

    if pattern_range is None:
        pattern_range = get_vrange(exp)
        
    pmin, pmax = get_vrange_pos(exp, stretch=3.0)               
    ax1, ax2, ax3 = axes
    exp_plt= ax1.imshow(exp, zorder=-10, vmin=pmin, vmax=pmax)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(exp_plt, cax=cax1, orientation='vertical')

    pmin, pmax = get_vrange_pos(sim, stretch=3.0)  
    sim_plt=ax2.imshow(sim, alpha=1.0, vmin=pmin, vmax=pmax)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sim_plt, cax=cax2, orientation='vertical')

    ratio=(exp/np.ma.masked_less_equal(sim, 0))
    #pmin, pmax = get_vrange(ratio, stretch=3.0) 
    #
    
    if ratio_range is None:
        asy_plt=ax3.imshow(ratio, cmap="terrain")
    else:
        pmin, pmax = ratio_range
        asy_plt=ax3.imshow(ratio, cmap="gnuplot2", vmin=pmin, vmax=pmax)
        
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(asy_plt, cax=cax3, orientation='vertical')

    for ax in axes:
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

    plt.draw()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    return