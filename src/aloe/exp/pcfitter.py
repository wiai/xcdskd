import numpy as np
import numpy.testing as npt
import math
from scipy import stats
from sklearn import linear_model
import cv2
from matplotlib import pyplot as plt


import logging
from ..sys import aloe_logging
logger = logging.getLogger(__name__)


from ..math.euler import Rx,Rz,Ry
#from xcdsebsd.crystal import ebsdconst
from ..exp import calibpc 
from ..plotting import pcplotter
from ..plotting.make2Dmap import make2Dmap

import h5py

def fit_hyper(Q):
    """
    general fit of hyperplanes to list of points
    
    References:
    -----------
    Gander& Hrebicek, "Solving Problems in Scientific Computing", 3rd ed.
    Chapter 6, p. 97
    
    Input:
    ------
    
    Q[rows,cols] = XYZ[npoints,ndims]
    
    Notes:
    ------
    first rows contain hyperplane basis vectors
    last row of V contains "normal" vector
    in 2d this should hold:
    V[0] x V[1] = V[2]
    
    L=p+c*v1 (v1=V[0]) is least square line through all data points
    
    """
    #p = Q.mean(axis=0)
    p = stats.trim_mean(Q, 0.1, axis=0)
    Qt = Q - p[np.newaxis,:]
    [U, S, V] = np.linalg.svd(Qt,0)
    return p,V

    
    
class PCFitter(object):
    
    def __init__(self,gridw=200,gridh=150):
        self.yroll_rad=0.0
        self.zroll_rad=0.0
        self.xtilt_rad=0.0
        self.gridw=gridw
        self.gridh=gridh
        self.workdir=None
        return
        
    def load_test_data(self,beamfile,coordsfile):
        """
        test data
        """
        beams =np.loadtxt(beamfile)
        coords=np.loadtxt(coordsfile)
        self.set_data_brkr(beams,coords)
        return
        
        
    def set_data_brkr(self,beam3d,pc3d,top_clip=0.0):
        """ 
        set PC data 
        from beam indices  (3d, IX,IY,'1') homogeneous coordinates 
        and pc coordinates (3d, PCX,PCY,DD) in Bruker convention
        """
        self.BI=beam3d
        self.PC=calibpc.brkr_to_pcxyz(pc3d,top_clip=top_clip)
        logging.info('set_data_brkr using top_clip: '+str(top_clip))
        
        # convert from BRUKER-PC to microns in detector system
        #self.PC[:,0]=      self.PC[:,0] *ebsdconst.BRKR_WIDTH_MICRONS
        #self.PC[:,1]= (1.0-self.PC[:,1])*ebsdconst.BRKR_HEIGHT_MICRONS
        #self.PC[:,2]=-(    self.PC[:,2])*ebsdconst.BRKR_HEIGHT_MICRONS
        
        self.PC0=self.PC.mean(axis=0)
        print('Beam Indices IX min/max: ',np.min(self.BI[:,0]), np.max(self.BI[:,0]))
        print('Beam Indices IY min/max: ',np.min(self.BI[:,1]), np.max(self.BI[:,1]))
        print('Mean PC: \n', self.PC0,'\n')
        self.PC=self.PC-self.PC0
        
        return


    def fit_xtilt(self):
        """
        fit 2D plane normal for list of 3D points
        assuming only tilt in X
        yz-projection is a line
        
        Input:
        ------
        XYZ: point list XYZ[rows,cols] = XYZ[npoints,3]

        """
        
        y=self.PC[:,1].reshape((-1,1))
        z=self.PC[:,2].reshape((-1,1))
        
        # Fit line using all data
        model = linear_model.LinearRegression()
        model.fit(z, y)

        # Robustly fit linear model with RANSAC algorithm
        #model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac = linear_model.RANSACRegressor()
        model_theilsen = linear_model.TheilSenRegressor()
        model_ransac.fit(z, y)
        model_theilsen.fit(z, y)
        inlier_mask = model_ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        # Predict data of estimated models
        line_X = np.arange(np.min(z), np.max(z))
        line_y = model.predict(line_X[:, np.newaxis])
        line_y_ransac   = model_ransac.predict(line_X[:, np.newaxis])
        line_y_theilsen = model_theilsen.predict(line_X[:, np.newaxis])
        
        # Compare estimated coefficients
        print("Estimated coefficients (normal, RANSAC, TheilSen):")
        print(model.coef_, model_ransac.estimator_.coef_,model_theilsen.coef_)
       
        print('X-Tilt LR       :', np.degrees(np.arctan(model.coef_)-0.5*math.pi))
        print('X-Tilt RANSAC   :', np.degrees(np.arctan(model_ransac.estimator_.coef_)-0.5*math.pi))
        print('X-Tilt Theil-Sen:', np.degrees(np.arctan(model_theilsen.coef_)-0.5*math.pi))
        
        
        lw=2
        #plt.scatter(z, y, color='yellowgreen', marker='.',
        #    label='yz projected data')
        plt.scatter(z[inlier_mask], y[inlier_mask], color='black', marker='.',
            label='Inliers')
        plt.scatter(z[outlier_mask], y[outlier_mask], color='red', marker='.',
            label='Outliers')    
        
        plt.plot(line_X, line_y, color='navy', linestyle='-', linewidth=lw,
         label='Linear regressor')
        plt.plot(line_X, line_y_ransac, color='cornflowerblue', linestyle='-',
         linewidth=lw, label='RANSAC regressor')
        plt.plot(line_X, line_y_theilsen, color='skyblue', linestyle='-',
         linewidth=lw, label='Theil-Sen regressor') 
        plt.legend(loc='lower right')    
        plt.gca().invert_xaxis()
        plt.axes().set_aspect('equal', 'datalim')
        if self.workdir is None:
            plot_filename='PCYZ_LINE.png'
        else:
            plot_filename=self.workdir+'/PCYZ_LINE.png'
        logging.info('saving regressor fits: '+plot_filename)
        plt.savefig(plot_filename,dpi=300,bbox_inches = 'tight')
        #plt.show()
        plt.close()
        
        return

        
    def fit_plane(self):
        """
        fit 2D plane normal (v3) for list of 3D points
        
        Input:
        ------
        XYZ: point list XYZ[rows,cols] = XYZ[npoints,3]

        """
        logging.info('fit_plane,  SVD hyperplane fit')
        # hyperplane fit
        self.p,V=fit_hyper(self.PC)
        self.v1=V[0]#/np.linalg.norm(V[0])
        self.v2=V[1]#/np.linalg.norm(V[1])
        self.v3=V[2]#/np.linalg.norm(V[2])
        
        #n=np.cross(v1,v2)
        #n=n/np.linalg.norm(n)

        # check handedness of the 3 vector coordinate system
        DV = np.linalg.det(V)
        print('SVD Determinant: ', DV)
        logging.info('Determinant of SVD V: %g' % DV)
        
        print('Hyperplane fit:')
        print(self.v1)
        print(self.v2)
        print(self.v3) # normal vector
        print(self.p)

        # v3: best fit sample plane normal S unit vector coordinates in DETECTOR (SCREEN) SYSTEM
        Sabs=np.linalg.norm(V[2]) # should be 1.0
        Sx=self.v3[0]/Sabs; Sy=self.v3[1]/Sabs; Sz=self.v3[2]/Sabs;
        if (Sz<0): # make normal point towards screen
            Sx=-Sx; Sy=-Sy; Sz=-Sz;
        S0=np.asmatrix([Sx,Sy,Sz]) 
        
        print('Sample normal SVD fit:')
        print( '%g %g %g' % (Sx,Sy,Sz)  )
        print(S0)
        logging.info('PCFITTER: Sample normal SVD fit %g %g %g' % (Sx,Sy,Sz))
        
        
        # 2-angle model for orientation of the best fit plane normal    
        # 1. z-rot of detector-x: roll around z to tilt X-axis away from horizontal 
        # if sample perfectly mounted and stage-X-axis aligned with detector X-axis, this should be near ZERO)
        # measured from y-axis: -0.5*math.pi    
        self.zroll_rad=np.arctan2(Sy,Sx)-0.5*math.pi 
        # 2. tilt around new non-horizontal X axis out of detector plane (near -20 deg)
        # tan(xtilt)=rxy/Sz
        # need in-screen-plane projection of plane normal (=new z)
        #rxy=math.sqrt(Sx*Sx+Sy*Sy)
        #self.xtilt_rad=-math.atan2(rxy,Sz)
        
        
        # alternative: Z of normal unit vector is cos of tilt angle
        # sign of angle depends on y-coordinate of sample normale
        # for positive Sy we have _negative_ tilting
        #if Sy>=0 : 
        #    self.xtilt_rad=-np.arccos(Sz)
        #else:
        #    self.xtilt_rad=+np.arccos(Sz)
        self.xtilt_rad=-np.arccos(Sz)
            
        print('hyperplane SVD Fit: (1st) zROLL (deg):',np.degrees(self.zroll_rad))
        print('hyperplane SVD Fit: (2nd) xTILT (deg):',np.degrees(self.xtilt_rad))
        # X-TILT angle is always positive in this convention 
        # actual direction governed by previous zROLL
        
        # TODO
        # the definition with z-x tilt is experimentally awkward:
        # procedure: start with sample aligned with detector system axes
        # the several steps to move to final alignment of axes
        # 1. zroll to describe non-horizontal X-tilt axis (can be small and fixed for fixed microscope!)
        # 2. tilt around X
        # 3. rotate around y to describe left-right roll of sample (mounting, changes for each sample)
        # 4. rotate around z to describe in-plane sample rotation (can be 0)
        # fitted sample normal only gives effect of 2 and (1,3) combined 
        
        
        # FIX scan-rot and x-axis tilt position: 
        # we can assume that the x-tilt-axis is aligned with the scan
        # observe horizontal line in SEM with different tilts
        # make sure that scan is along X-axis in final scan point image
        # -> this assures that scanX || tiltX and thus also defines the sampleX
        # sampleX is defined to be along tiltX because it can be aligned this way!
        #
        
        #TODO: all into yroll
        #self.zroll_rad=0.0;
        # for y-roll think backwards: xtilt has to be different!
        #rxs=math.sqrt(Sx*Sx+Sz*Sz)
        #self.yroll_rad=np.arctan2(Sx,Sz)
        #if Sy>=0: 
        #    self.xtilt_rad=-np.arccos(rxs)
        #else:
        #    self.xtilt_rad=+np.arccos(rxs)

        
        print('hyperplane SVD Fit Z-X-Y-Roll:')
        print('------------------------------------------------------')
        print('(1) zROLL (deg):',np.degrees(self.zroll_rad))
        print('(2) xTILT (deg):',np.degrees(self.xtilt_rad))        
        print('------------------------------------------------------')  
        
        # check that rotation of 001 gives surface normal in detector system:
        Stest=np.asmatrix([0,0,1])*Rx(self.xtilt_rad)*Rz(self.zroll_rad)
        print("check S     :", Stest)
        print("check S-S0 :", Stest-S0)
        npt.assert_almost_equal(Stest, S0, decimal=8)
        return
        
    def fit_affine(self,X,Y,plotdir=''):
        """ least square affine transformation fit
            
        Parameters:
        -----------
        X : beam indices 3D
        Y : pc values 3D

        Notes:
        ------
        original code example from
        http://stackoverflow.com/questions/20546182/how-to-perform-coordinates-affine-transformation-using-python-part-2
        we only need 3D, not 4D as in example
        """

        
        print('Least Square Projective Transformation Fit of PC...')
    
        pts_src=X[:,0:2].reshape(-1,1,2)
        pts_dst=Y[:,0:2].reshape(-1,1,2)
    
    
        print(pts_src)
        print(pts_dst)
    
        # Calculate Homography
        h, status = cv2.findHomography(pts_src, pts_dst)
        print('h \n',h) #status)
    
        #transform = lambda x: unpad(np.dot(pad(x), An))
        
        #A=AP[0:3,0:3]
        #A=An[0:3,0:3]
        
        fit=np.dot(X,h.T)
        #fit=transform(primary)
        
        
        print("Transformation Homography Matrix h:\n", h)
        print("Max error of projective fit:", np.abs(Y - fit).max())

        XTilt=plotPC(Y,fit,plotdir=plotdir)

        #print ('COMPARE TO FREE MATRIX:')
        #A2=np.array([[ -6.49412890e-04,  -8.31804431e-05,  -5.20287978e-05],
        #             [ -3.63551245e-06,  -7.87646129e-04,   4.53066977e-04],
        #             [  5.04812094e-01,   2.07732094e-01,   8.28988425e-01]])
        #fit2=np.dot(X,A2)
        #plotPC(fit2,fit,plotdir=plotdir)
        #print("Max error between fits:", np.abs(fit2 - fit).max())

        
        return 


    def fit_projective(self):
        logging.info('fit_projective, projective transformation PC fit')

        self.plane_PC=np.squeeze((self.PC-self.p)@Rx(-self.xtilt_rad)@Rz(-self.zroll_rad))
        
        self.minv1=np.min(self.plane_PC[:,0])
        self.minv2=np.min(self.plane_PC[:,1])
        
        self.maxv1=np.max(self.plane_PC[:,0])
        self.maxv2=np.max(self.plane_PC[:,1])
        
        self.plane_quad=self.p+ np.asmatrix([[self.minv1,self.minv2,0],
                         [self.maxv1,self.minv2,0],
                         [self.maxv1,self.maxv2,0],
                         [self.minv1,self.maxv2,0]])@Rx(-self.xtilt_rad)@Rz(-self.zroll_rad)
        print('Best fit plane quad:')
        print(self.plane_quad)
        self.plane_quad_3d=np.copy(self.plane_quad)
        # switch y and z for plotting
        #self.plane_quad_3d[:,1]=np.copy(self.plane_quad[:,2])[0]
        #self.plane_quad_3d[:,2]=np.copy(self.plane_quad[:,1])[0]
        self.plane_quad_3d[:,[1, 2]] = self.plane_quad_3d[:,[2, 1]]
        print('Best fit plane quad 3d:')
        print(self.plane_quad_3d)
        
        # now fit 3x3 projective transformation in the best fit plane
        self.h=self.get_homography(self.BI,self.plane_PC)
        print('homography: \n',self.h)
        #np.savetxt('h_fitted.dat',self.h)
        
        # find projected points in plane
        
        """
        self.h=np.array( [  [ -6.64775265e+00,   9.81448782e-01,   1.89711372e+03],
                            [  1.93715608e-01,   2.04577080e+01,  -4.78265414e+03],
                            [ -2.39175628e-06,   5.04737285e-04,   1.00000000e+00]])
        
        self.h=np.array( [  [ -hstep,           h_shear_yconst,   h_trans=x_scan_pos(0,0)=+mw/2],
                            [  v_shear_xconst,          v_step,   v_trans=y_scan_pos(0,0)=-mh/2],
                            [  h_trapez,              v_trapez,        1.00000000e+00    ]])                    
                            
        
        self.h=np.array( [  [ -10.0,  2.0,   3200],
                            [  0.0,  10.0,  -2400],
                            [  0.0 ,  0.0,   1.00]])
        print('adjusted homography: \n',self.h)
        """
        # additional z-rotation to align x-axes
        scan_cell=self.project_grid(2,2)
        print(scan_cell)
        p0=scan_cell[0]
        pX=p0-scan_cell[1]
        pY=p0-scan_cell[2]
        pXl=np.linalg.norm(pX)
        pYl=np.linalg.norm(pY)
        print('projected X-scan vectors:',pX,pXl)
        print('projected Y-scan vectors:',pY,pYl)
        print('step size ratio Y/X: ',pYl/pXl)
        self.x_map_angle=math.atan2(pX[1],pX[0])
        print('Map Rotation to Align X:', np.degrees(self.x_map_angle))
        
        # pc coordinates projected from beam indices and tilt parameters
        #print(self.BI)
        self.proj_PC=calibpc.project_points(self.BI,self.h)@Rx(self.xtilt_rad)@Rz(self.zroll_rad)
        #print(self.proj_PC)
        #self.inplanexy=calibpc.project_points(self.BI,self.h)
        self.proj_PC_brkr=self.project_to_brkr(self.proj_PC+self.PC0)
        return
    
    def project_idx(self,BI):
        xyz=calibpc.project_points(BI,self.h)
        return xyz[:,0:2]
        
    def project_grid(self,width,height):
        """ make the 2D in-plane grid projection"""
        grid_points=np.array(calibpc.make_map_indices(width,height)).T
        xyz=calibpc.project_points(grid_points,self.h)
        self.grid2D=xyz[:,0:2]
        return xyz[:,0:2]
    
    def get_homography(self,BI,PC):
        """ get best-fit sub Homography for >>2D<< xy point sets out of 3D PC data
        """
        pts_src=BI[:,0:2] # x y scan indices 
        pts_dst=np.array([PC[:,0],PC[:,1]]).T
        h, status = cv2.findHomography(pts_src, pts_dst, cv2.LMEDS) # (cv2.RANSAC,err) cv2.LMEDS, 0
        return h  
    
    def plot_fitted_data(self,outdir=None,show=False):
        logging.info('calling pcplotter, plane_PC shape: ' + str(self.plane_PC.shape))
        self.plotter=pcplotter.PCPlotter(self.PC,self.proj_PC,grid2D=self.grid2D,plane_PC=self.plane_PC,BI=self.BI)
        if outdir is not None:
            self.plotter.plot(show=show,plotdir=outdir)
        return
        
    def project_to_brkr(self,pcxyz,top_clip=0.0):
        print('projecting to Bruker PC parameters...')
        #print(pcxyz)
        pcbrkr=calibpc.pcxyz_to_brkr(pcxyz, top_clip=top_clip).T
        #print('PCX,PCY,DD:')
        #print(pcbrkr)
        #print(pcbrkr.shape)
        return pcbrkr
        
    def save_pc_maps(self,w,h,filename,top_clip=0.0,h5appendfile=None):
        logging.info('saving PC maps using top_clip: '+str(top_clip))
        # beam positions in scan grid and z=1
        ix,iy,iz = calibpc.make_map_indices(w,h)
        beams=np.array([ix,iy,iz]).T
        # absolute pc coordinates in detector system
        pcxyz=self.PC0+calibpc.project_points(beams,self.h)@Rx(self.xtilt_rad)@Rz(self.zroll_rad)
        # scan points in sample plane
        scanxy=calibpc.project_points(beams,self.h)
        
        # maps of scan positions
        x_scan=make2Dmap(scanxy[:,0],beams[:,0],beams[:,1],h,w)
        y_scan=make2Dmap(scanxy[:,1],beams[:,0],beams[:,1],h,w)
        np.savetxt(filename+'_XSCAN.MAP',x_scan)
        np.savetxt(filename+'_YSCAN.MAP',y_scan)
        
        # maps of projective PC-fit in microns
        x_pc=make2Dmap(pcxyz[:,0],beams[:,0],beams[:,1],h,w)
        y_pc=make2Dmap(pcxyz[:,1],beams[:,0],beams[:,1],h,w)
        z_pc=make2Dmap(pcxyz[:,2],beams[:,0],beams[:,1],h,w)
        np.savetxt(filename+'_XPC.MAP',x_pc)
        np.savetxt(filename+'_YPC.MAP',y_pc)
        np.savetxt(filename+'_ZPC.MAP',z_pc)        
        
        # maps of Bruker parameters for pattern calibration
        pcbrkr=self.project_to_brkr(pcxyz, top_clip=top_clip)
        pcx=make2Dmap(pcbrkr[:,0],beams[:,0],beams[:,1],h,w)
        pcy=make2Dmap(pcbrkr[:,1],beams[:,0],beams[:,1],h,w)
        ddd=make2Dmap(pcbrkr[:,2],beams[:,0],beams[:,1],h,w)
        np.savetxt(filename+'_PCXP.MAP',pcx, fmt='%10.6f')
        np.savetxt(filename+'_PCYP.MAP',pcy, fmt='%10.6f')
        np.savetxt(filename+'_DDP.MAP' ,ddd, fmt='%10.6f')
        
        
        #np.savetxt('pcfit_exp_PC.dat',self.PC)
        #np.savetxt('pcfit_inplane_exp_PC.dat',self.plane_PC)
        
        
        # hdf5
        print('HDF5 output...')
        print(self.grid2D)
        #if h5appendfile:
        #   print("Appending PCFITTER data to:", h5appendfile)
        #    f = h5py.File(h5appendfile, "a")
        #else:
        
        f = h5py.File(filename+'_PC_XCDS.hdf5', "w")
            
        f["PCFITTER/PC_XCDS_FIT"]=self.PC
        f["PCFITTER/PC_PROJ_MAP_FIT"]=self.proj_PC
        f["PCFITTER/PROJ_MAPGRID_SAMPLE"]=self.grid2D
        f["PCFITTER/PROJ_POINTS_SAMPLE"]=self.plane_PC
        f["PCFITTER/BEAMIDX"]=self.BI
        f["PCFITTER/BRKR_PC"]=pcbrkr
        f["PCFITTER/BRKR_PC_PCX_MAP"]=pcx
        f["PCFITTER/BRKR_PC_PCY_MAP"]=pcy
        f["PCFITTER/BRKR_PC_DDD_MAP"]=ddd
        f.close() 
        
        return
        

def run_pcfitter():
    fitter=PCFitter()
    fitter.load_test_data("test_data/beam_indices.txt","test_data/pc_coords.txt")
    fitter.fit_plane()
    fitter.fit_projective()
    fitter.plot_fitted_data()
    
    #fitter.plot(fitter.plane_PC[::4])
    return

   
if __name__=='__main__':
    run_pcfitter()

   