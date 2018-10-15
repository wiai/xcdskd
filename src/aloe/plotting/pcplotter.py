import numpy as np
from scipy import stats

import matplotlib
#matplotlib.use("Qt5Agg")   # use PyQt5
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.ticker import MultipleLocator
from matplotlib import rc

import os


def makeLabelList2D(bracket1,bracket2,IndexArray):
    '''
    returns the 2D indices formated with chosen bracket
    '''    
    LabelList=list()
    for ix in range(IndexArray.shape[1]):
        #print(IndexArray[:,ix])
        formattedline =(bracket1+'%i,%i'+bracket2) % ( tuple(IndexArray[:,ix]) )
        LabelList.append(formattedline)

    return np.array(LabelList)




class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)




class PCPlotter(object):  
    
    def __init__(self,primary,secondary=None,plane_quad=None,
                    grid2D=None,plane_PC=None,BI=None):
        #primary plot coordinate list
        self.primary=primary
        # secondary data for comparison
        if secondary is not None:
            self.secondary = secondary
        else:
            self.secondary =None
        # best fit plane vertices    
        if plane_quad is not None:
            self.plane_quad = plane_quad
        else:
            self.plane_quad = None
        # best fit in plane: all grid points    
        if grid2D is not None:
            self.grid2D = grid2D
        else:
            self.grid2D = None
        # in-plane experimental scan points from PC fit  
        if plane_PC is not None:
            self.plane_PC = plane_PC
        else:
            self.plane_PC = None 
        # beam indices for PC value list 
        if BI is not None:
            self.BI = BI
        else:
            self.BI = None             
        self.PLOT_SCAN_LABELS=True    
   

   
    def plot(self,plotdir='.', show=False, plot3d=False):
        ''' plot projection centers
        '''
        if not os.path.exists(plotdir):
            print('pcplotter, creating dir: ',plotdir)
            os.makedirs(plotdir)
        
        
        primary=self.primary
        if self.secondary is not None:
           secondary=self.secondary
        
        if self.secondary is not None:
            error_vecs= self.secondary - self.primary
            #errors=np.ravel(np.abs(error_vecs[:,1])) # check xyz
            errors=np.linalg.norm( error_vecs, axis=1)
            #print(errors)  
            
            #errors=np.mean(np.abs(secondary[:,0] - primary[:,0]))
            #print("Mean error of projective primary:", np.mean(errors))
            print("Median error of projective fit:", np.median(errors))
            fig = plt.figure()
            plt.hist(errors,bins=71)
            plt.title('projective geometry model: estimation of single-pattern error\n'
                +'median of errors={0: >#05.2f} $\mu m$'. format(float(np.median(errors))))
            plt.ylabel('number of measurements', color='k', fontsize=22)
            plt.xlabel(r'$|PC_{\mathrm{exp}}-PC_{\mathrm{projective}}|$ ($\mu m$)', color='k',fontsize=22)    
            plt.savefig(plotdir+'PCXYZ_ERRORS.png',dpi=300,bbox_inches = 'tight')
            if show:
                plt.show()
            plt.close()

        if (self.grid2D is not None):
            # this plot is in the SAMPLE SYSTEM
            # (x axis opposite direction to Detector-X)
            X_AXIS_FACTOR=-1.0
            fig, ax = plt.subplots(1,1)
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)    
            #plt.title(title,fontsize=26)
            ax.set_ylabel(r'Y$_{SAMPLE}$ ($\mu m$)', color='k', fontsize=22)
            ax.set_xlabel(r'X$_{SAMPLE}$ ($\mu m$)', color='k',fontsize=22)

            ax.scatter(X_AXIS_FACTOR*self.grid2D.T[0],self.grid2D.T[1],s=8,c='lightblue',alpha=0.9, lw=0)
            # experimental points
            if self.plane_PC is not None:
                #print(self.plane_PC.T[0], self.plane_PC.T[1])
                ax.scatter(X_AXIS_FACTOR*self.plane_PC.T[0], self.plane_PC.T[1], s=5, c='k', alpha=0.9, lw=0)
            # labels
            if self.PLOT_SCAN_LABELS:
                labels = makeLabelList2D('(',')',self.BI.T[0:2])
                # todo: logging
                #print(labels)
                #print(self.plane_PC.T[0], self.plane_PC.T[1])
                for label, x, y in zip(labels, np.ravel(X_AXIS_FACTOR*self.plane_PC.T[0]), np.ravel(self.plane_PC.T[1])):
                    ax.annotate(
                        label, zorder=105,
                        xy = (x, y), xytext = (0, 10), fontsize=6,
                        color='black', #weight='bold',
                        textcoords = 'offset points', ha = 'center', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.6),
                        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))            
                
                
            ax.grid(True)
            ax.invert_xaxis()
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_aspect('equal', 'datalim')
            plt.savefig(plotdir+'SCAN_SAMPLE.png',dpi=300,bbox_inches = 'tight')
            if show:
                plt.show()
            plt.close()    
            
            
        fig, ax = plt.subplots(1,1)
        #ax = fig.add_subplot(221)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)    
        #plt.title(title,fontsize=26)
        ax.set_ylabel(r'$\Delta$PC$_Y$ ($\mu m$)', color='k', fontsize=22)
        ax.set_xlabel(r'$\Delta$PC$_X$ ($\mu m$)', color='k',fontsize=22)
        ax.scatter(primary.T[0],primary.T[1],s=12,c='r',alpha=0.9, lw=0)
        if self.secondary is not None:
            ax.scatter(secondary.T[0],secondary.T[1],s=5,c='b',alpha=0.9,lw=0)
        ax.grid(True)
        ax.set_aspect('equal', 'datalim')
        plt.savefig(plotdir+'PCXY.png',dpi=300,bbox_inches = 'tight')
        if show:
            plt.show()
        plt.close()
       


        fig, ax = plt.subplots(1,1)
        #ax = fig.add_subplot(222)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)    
        #plt.title(title,fontsize=26)
        ax.set_ylabel(r'$\Delta$PC$_Z$ ($\mu m$)', color='k', fontsize=22)
        ax.set_xlabel(r'$\Delta$PC$_X$ ($\mu m$)', color='k',fontsize=22)
       
        ax.scatter(primary.T[0],primary.T[2],s=22,c='red',alpha=1.0,lw=0)
        if self.secondary is not None:
            plt.scatter(secondary.T[0],secondary.T[2],s=10,c='blue',alpha=1.0,lw=0)
        ax.grid(True)
        ax.set_aspect('equal', 'datalim')
        ax.invert_yaxis()
        plt.savefig(plotdir+'PCXZ.png',dpi=300,bbox_inches = 'tight')
        if show:
            plt.show()
        plt.close()
       
        fig, ax = plt.subplots(1,1)
        #ax = fig.add_subplot(223)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)    
        #plt.title(title,fontsize=26)
        ax.set_ylabel(r'$\Delta$PC$_Y$ ($\mu m$)', color='k', fontsize=22)
        ax.set_xlabel(r'$\Delta$PC$_Z$ ($\mu m$)', color='k',fontsize=22)
        #plt.gca().invert_yaxis() # consistent with y measured from top of pattern
        ax.scatter(primary.T[2],primary.T[1],s=22,c='r',alpha=1.0,lw=0)
        if self.secondary is not None:
            ax.scatter(secondary.T[2],secondary.T[1],s=10,c='b',alpha=1.0,lw=0)
        #plt.plot(primary.T[2],fittedY,'y-' )
        ax.grid(True)
        ax.set_aspect('equal', 'datalim')
        ax.invert_xaxis()
        plt.savefig(plotdir+'PCYZ.png',dpi=300,bbox_inches = 'tight')
        if show:
            plt.show()
        plt.close()
        
        
        # ---------- 3D  PLOT ----------------
        if plot3d:
            rc('font',size=16)
            #rc('font',family='serif')
            rc('axes',labelsize=16)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
            # best fit PLANE
            if self.plane_quad is not None:
                quad_verts=[self.plane_quad_3d]
                ax.add_collection3d( Poly3DCollection(quad_verts, facecolors='b', linewidths=1, alpha=0.4) )
        
            ax.scatter(primary.T[0],primary.T[2],primary.T[1], c='r', s=8, marker='o',lw=0.1, edgecolors='k',alpha=1.0)
            plt.grid(True)
        
            #if self.secondary is not None:
            #ax.scatter(secondary.T[0],secondary.T[2],secondary.T[1],s=16, c='b', marker='o',lw=0,alpha=0.3)

            ax.set_xlabel('\n'+r'$\Delta$PC$_X$  ($\mu m$)')
            ax.set_ylabel('\n\n'+r'$\Delta$PC$_Z$  ($\mu m$)')
            ax.set_zlabel('\n\n'+r'$\Delta$PC$_Y$  ($\mu m$)')
            #plt.title(r'sample plane tilt relative to detector plane'+'\n'
            #    +r'$\tau_X$={0: >#05.2f}'. format(float(np.degrees(self.xtilt_rad)))
            #    +r'°   $\nu_Z$= {0: >#05.2f}'. format(float(np.degrees(self.ztilt_rad)))+'° \n' )
            min3d=np.min(primary)
            max3d=np.max(primary)
            ax.set_xlim3d(min3d, max3d)
            ax.set_ylim3d(min3d, max3d)
            ax.set_zlim3d(min3d, max3d)
        
            [t.set_va('center') for t in ax.get_yticklabels()]
            [t.set_ha('left') for t in ax.get_yticklabels()]
            [t.set_va('center') for t in ax.get_xticklabels()]
            [t.set_ha('right') for t in ax.get_xticklabels()]
            [t.set_va('center') for t in ax.get_zticklabels()]
            [t.set_ha('left') for t in ax.get_zticklabels()]
        
            ax.xaxis._axinfo['tick']['inward_factor'] = 0
            ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
            ax.yaxis._axinfo['tick']['inward_factor'] = 0
            ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
            ax.zaxis._axinfo['tick']['inward_factor'] = 0
            ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
            ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
        
            ax.xaxis.set_major_locator(MultipleLocator(3000))
            ax.yaxis.set_major_locator(MultipleLocator(3000))
            ax.zaxis.set_major_locator(MultipleLocator(3000))
        
            plt.gca().invert_yaxis() # view from detector along negative Z
            plt.savefig(plotdir+'PCXYZ_FIT.png',dpi=300) # ,bbox_inches = 'tight')
            if show:
                plt.show()
            plt.close()  