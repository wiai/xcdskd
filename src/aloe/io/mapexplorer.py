"""
Matplotlib-based EBSD map explorer: point to map point and show pattern
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


import ipywidgets as w
import warnings
from skimage.io import imsave
from aloe.image.utils import img_to_uint

class MapExplorer():
    
    def __init__(self, ebsd, basemap):
        self.ebsd = ebsd
        self.basemap = basemap
        
    def init_widgets(self):
        self.savebutton = w.Button(
            description='Save Pattern',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save current EBSD pattern',
            icon=''
        )
        self.savebutton.on_click(self.on_savebutton_clicked)
        
        self.ix_slider = w.IntSlider(continuous_update=True, orientation='horizontal', max=self.ebsd.map_width-1)
        self.iy_slider = w.IntSlider(continuous_update=True, orientation='horizontal', max=self.ebsd.map_height-1)
        self.neighbor_slider = w.IntSlider(continuous_update=False, orientation='horizontal', max=10)

        self.ix_slider.value = self.ebsd.map_width // 2
        self.iy_slider.value = self.ebsd.map_height // 2

        self.sliders = [self.ix_slider, self.iy_slider, self.neighbor_slider]
        for s in self.sliders:
            s.observe(self.update_inspector)
        

    def on_savebutton_clicked(self,b):
        #print("Save Button clicked.")
        ix = self.ix_slider.value
        iy = self.iy_slider.value
        nap = self.neighbor_slider.value
        fname = 'nap_'+str(iy)+'_'+str(ix)+'_'+str(nap)+'.tif'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(fname=fname, arr=img_to_uint(self.current_pattern, dtype=np.uint16), plugin="tifffile")
            #np.savetxt("current_pattern.dat", self.current_pattern)
        
    def init_plot(self):
        plt.rcParams["figure.figsize"] = [8,3]
        self.fig = plt.figure()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        self.ax_map = ax1.imshow(self.basemap, alpha=1.0, cmap="viridis")
        self.ebsd_pattern = ax2.imshow(self.ebsd.get_nap(0,0), alpha=1.0)#, vmin=-0.02, vmax=0.02, cmap=plt.get_cmap('binary'))
        self.circ = Circle((5, 5), 3.0, color='b', fill=False)
        ax1.add_artist(self.circ)
        self.circ_move = Circle((5, 5), 3.0, color='y', fill=False)
        ax1.add_artist(self.circ_move)
        plt.show()
        self.map_stepx = 1 # step size factor in SEM image vs. EBSD pattern scan
        self.map_stepy = 1
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_leave = self.fig.canvas.mpl_connect('axes_leave_event', self.on_leave_ax)
        
    def on_mouse_move(self, event):
        if(event.inaxes):
            ix = int(event.xdata)
            iy = int(event.ydata)
            #print("(x,y): ", round(event.xdata), "\t", round(event.ydata))
            self.circ_move.center = ix*self.map_stepx, iy*self.map_stepy
            
            save_nap = self.ebsd.nap
            self.ebsd.nap = 0
            self.current_pattern = self.ebsd.get_nap(ix, iy, invert=False)
            #self.current_pattern = self.ebsd.get_pattern_data(ix, iy)['pattern']
            self.ebsd_pattern.set_data(self.current_pattern)
            self.ebsd.nap = save_nap
        
    def on_click(self, event):
        ix = int(event.xdata)
        iy = int(event.ydata)
        self.circ.center = ix*self.map_stepx, iy*self.map_stepy
        self.current_pattern = self.ebsd.get_nap(ix, iy, invert=False)
        self.ebsd_pattern.set_data(self.current_pattern)
        self.ix_slider.value = ix
        self.iy_slider.value = iy
        
    def on_leave_ax(self, event):    
        self.update_inspector(None) # reset to current values on sliders

    def update_inspector(self, change):
        ix = self.ix_slider.value
        iy = self.iy_slider.value
        self.circ.center = ix*self.map_stepx, iy*self.map_stepy
        self.circ_move.center = ix*self.map_stepx, iy*self.map_stepy
        self.ebsd.nap = self.neighbor_slider.value
        self.current_pattern = self.ebsd.get_nap(ix, iy, invert=False)
        self.ebsd_pattern.set_data(self.current_pattern)
   
    def show_widgets(self):
        self.update_inspector(None)
        display(w.HBox( [w.VBox([self.ix_slider, self.iy_slider, self.neighbor_slider]), 
                w.VBox([self.savebutton]) ]))