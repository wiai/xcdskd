import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

def plotmap(mapgrid, settings={}):
    """
    plotting maps of real values
    """
    # masking out zero values of mask array
    mask = settings.get("mask", None)
    if mask is not None:
        mapgrid = np.ma.masked_where(mask==0, mapgrid)
    
    # save plot to filename if given
    save_filename = settings.get("save_filename", None)
    
    nrows, ncols = mapgrid.shape
    stepsize = settings.get("stepsize", None)
    stepsize_unit = settings.get("stepsize_unit", " ($\mu$m)")

    # default size of width and height of map
    # map pixels outer limits are assumed to extend from 0...ncols, 0...nrows
    # TODO: make sure that matplotlib plotting places 0.5*microns,0.5*microns at first pixel center
    map_spatial_width = 1.0
    map_spatial_height = 1.0
    if stepsize is not None:
        map_spatial_width = ncols * stepsize
        map_spatial_height = nrows * stepsize
            
    # Alignment of the spatial oordinate system axes relative to the scan grid
    # 0 => x_spatial = +col, y_spatial = -row
    # 1 => x_spatial = +row, y_spatial = +col
    # 2 => x_spatial = -col, y_spatial = +row (this is the most common setup, assumed as default)
    # 3 => x_spatial = -row, y_spatial = -col
    spatial_alignment = settings.get("spatial_alignment", 2)
    if spatial_alignment not in [0,1,2,3]:
        print("WARNING: incorrect option for spatial alignment: ", spatial_aligmnent)
        spatial_aligment == 2
        print("INFO   : setting default spatial alignment: ", spatial_aligmnent)
        
    if spatial_alignment==0:
        col_spatial_label = "X1"
        row_spatial_label = "Y1"
        row_spatial_top = map_spatial_height
        row_spatial_bottom = 0.0
        col_spatial_left = 0.0
        col_spatial_right = map_spatial_width      
    elif spatial_alignment==1:
        col_spatial_label = "Y1"
        row_spatial_label = "X1"
        row_spatial_top = 0.0
        row_spatial_bottom = map_spatial_height
        col_spatial_left = 0.0
        col_spatial_right = map_spatial_width 
    elif spatial_alignment==3:
        row_spatial_top = map_spatial_height
        row_spatial_bottom = 0.0
        col_spatial_left = map_spatial_width 
        col_spatial_right = 0.0
        col_spatial_label = "Y1"
        row_spatial_label = "X1"
    else:
        # assume "2" default setting
        col_spatial_label = "X1"
        row_spatial_label = "Y1"
        row_spatial_top = 0.0
        row_spatial_bottom = map_spatial_height
        col_spatial_left = map_spatial_width
        col_spatial_right = 0.0
  
    cmap = settings.get("cmap", "inferno")
    colorbar_label = settings.get("colorbar_label", None)
    color_scan_labels = 'tab:gray'
    color_spatial_labels = 'k'
    colorbar_offset = 0.15 # fraction of map width
    cmin = settings.get("cmin", None)
    cmax = settings.get("cmax", None)
    
    fig, ax_scan = plt.subplots()

    ax_spatial_vertical = ax_scan.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax_spatial_vertical.set_ylabel('Y1 (' + stepsize_unit+')', color=color_spatial_labels)  # we already handled the x-label with ax1
    #ax_euler.plot(t, data2, color=color)
    ax_spatial_vertical.tick_params(axis='y', labelcolor=color_spatial_labels)
    ax_spatial_vertical.set_ylim(row_spatial_bottom, row_spatial_top)
    
    ax_spatial_horizontal = ax_spatial_vertical.twiny()  # instantiate a second axes that shares the same y-axis
    color = 'tab:blue'
    ax_spatial_horizontal.set_xlabel('X1 ('+ stepsize_unit+')', color=color_spatial_labels)  # we already handled the x-label with ax1
    ax_spatial_horizontal.tick_params(axis='x', labelcolor=color_spatial_labels)
    ax_spatial_horizontal.set_xlim(col_spatial_left, col_spatial_right)
    
    im = ax_scan.imshow(mapgrid, clim=[cmin, cmax], cmap=cmap)
    
    # colorbar
    cax = ax_scan.inset_axes([1.0 + colorbar_offset, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im, cax=cax, orientation='vertical', pad = 0.1)
    cb.set_label(colorbar_label, fontsize=12, labelpad=10)
    
    #ax_scan.set_aspect("equal")

    ax_scan.set_xlabel("scan grid column", color=color_scan_labels)  
    ax_scan.tick_params(axis='x', labelcolor=color_scan_labels)
    ax_scan.set_ylabel("scan grid row", color=color_scan_labels)  
    ax_scan.tick_params(axis='y', labelcolor=color_scan_labels)
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    return