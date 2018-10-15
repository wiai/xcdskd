import numpy as np
from scipy import stats
#from scipy import interpolate
from scipy.interpolate import Rbf

#import cv2


import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..exp import ebsdconst
#from ..cli import config_xcds 



def pcxyz_to_brkr(pc_xyz, top_clip=0.0):
    """ convert detector pc coordinates into bruker convention
    """
    image_width =ebsdconst.BRKR_WIDTH_MICRONS
    image_height=(1.0-top_clip)*ebsdconst.BRKR_HEIGHT_MICRONS
    
    pcx=      pc_xyz[:,0]/image_width
    pcy=1.0 - pc_xyz[:,1]/image_height
    dd =     np.abs(pc_xyz[:,2]/image_height)
    
    pc_brkr=np.array([pcx,pcy,dd])
    
    return pc_brkr
    
def brkr_to_pcxyz(pcbrkr,top_clip=0.0):
    """ convert Bruker pc coordinates PCX,PCY,DD into microns in detector system
    """
    image_width =ebsdconst.BRKR_WIDTH_MICRONS
    image_height=(1.0-top_clip)*ebsdconst.BRKR_HEIGHT_MICRONS
    
    x_microns= +image_width  *      pcbrkr[:,0]
    y_microns= +image_height * (1.0-pcbrkr[:,1])
    z_microns= -image_height *      pcbrkr[:,2]
    
    pc_xyz=np.array([x_microns,y_microns,z_microns]).T
    
    return pc_xyz    

    
    
def brkr_to_gnom(pcx,pcy,dd,aspect):
    """
    convert Bruker PCX,PCY,DD,Aspect to gnomonic
    """
    y_gn_max= + (    pcy)        /dd
    y_gn_min= - (1.0-pcy)        /dd
    x_gn_max= +((1.0-pcx)*aspect)/dd
    x_gn_min= -((    pcx)*aspect)/dd
    pc_gnom=np.array([x_gn_min,x_gn_max,y_gn_min,y_gn_max],dtype=np.float32)
    return pc_gnom
    

def pcxyz_to_gnom(pc_xyz, top_clip=0.0):
    """ convert detector pc coordinates to gnomonic projection values
    """
    image_width =ebsdconst.BRKR_WIDTH_MICRONS
    image_height=(1.0-top_clip)*ebsdconst.BRKR_HEIGHT_MICRONS
    
    # coordinates of image borders relative to PC
    x_min = - (               pc_xyz[:,0])  
    y_min = - (image_height - pc_xyz[:,1])  
    
    x_max =   (image_width  - pc_xyz[:,0]) 
    y_max =   (               pc_xyz[:,1])
    
    # gnomonic projection: scale by z
    # z is minus in pc_xyz
    img_xy =np.array([x_min,x_max,y_min,y_max])
    img_z = np.array(- pc_xyz[:,2])
    img_gnom = img_xy / img_z
    
    return img_gnom.T[0]    
    

    
    
    
def calibratePC(ScanPointList,bcf_filename,mapinfo,XTilt=-20.0):
    """
    This function calibrates the Projection Center from the current PC fit values
    in ScanPointList assuming that the step size and the sample tilt is known.
    This makes it possible to extrapolate all experimentally determined PC values
    to a common reference point (0,0) assuming a regular x-y grid of steps tilted by 
    around the detector X-axis.
    The extrapolated reference PC values are then averaged to result
    in the PC estimation.
    
       
    Assumptions: 
    ------------
    
    * the SEM beam X-scan is exactly parallel to the X-Tilt detector axis
    * Bruker: XTilt=(SampleTilt-90)-DetectorTilt (deg)
    * i.e. SampleTilt=70 usually leads to NEGATIVE XTilts!!!
    * the method produces PC values in a tilted rectangular region
    * no trapezoidal/projective distortion is considered
        
    """

    print('SCALING TRANSFORMATION PC Calibration...')
    print('Assuming pure XTilt, no trapez:', XTilt)

    f = open(bcf_filename+'_DSCALIBPC_LOG.TXT', "w")
       
    f.write('DynamicS Pattern Center calibration\n')
    
    StepSize=mapinfo['hstep']
    
    print("scanPointList: ", ScanPointList[0,:])
    ImageWidthPX =ScanPointList[0,17]
    ImageHeightPX=ScanPointList[0,18]
    f.write(str(ImageWidthPX)  + '    # PatternWIDTH\n')
    f.write(str(ImageHeightPX) + '    # PatternHEIGHT\n')
    alpha=-XTilt*np.pi/180.0 
    f.write(str(XTilt) + '    # total tilt angle between sample surface normal and direction to PC\n')
    #relative step size
    hStepRel=StepSize/ebsdconst.BRKR_WIDTH_MICRONS 
    
    # how many pixels moves the pattern center per scan point
    hStepPX= hStepRel*ImageWidthPX
    vStepPX= hStepPX*np.cos(alpha) # just tilted, no dependence of hStep on vStep
    zStepPX= hStepPX*np.sin(alpha)
    
    f.write('assumed PC beam steps in PX:\n')
    f.write(str(-hStepPX)+ '    # horizontal PC step size in pixels\n')
    f.write(str(vStepPX)+ '    # vertical   PC step size in pixels\n')
    f.write(str(zStepPX)+ '    # z          PC step size in pixels\n')
    
    #meanBX=np.round(np.mean(ScanPointList[:,15]))
    #meanBY=np.round(np.mean(ScanPointList[:,16]))
    meanBX=0.0
    meanBY=0.0
    f.write('Reference beam indices:\n')    
    f.write(str(meanBX)+ '\n')
    f.write(str(meanBY)+ '\n')
    dBX=ScanPointList[:,12]-meanBX
    dBY=ScanPointList[:,13]-meanBY
    
    # make absolute values in pixels
    # get away from 2 different units for PCX,PCY,DD given in Bruker software
    pPCX=      ScanPointList[:, 9]  * ImageWidthPX
    pPCY= (1.0-ScanPointList[:,10]) * ImageHeightPX
    pDD =      ScanPointList[:,11]  * ImageHeightPX
    
    # back-interpolate to mean reference PC
    pPCX0= np.mean(pPCX  + dBX*hStepPX)
    pPCY0= np.mean(pPCY  - dBY*vStepPX)
    pDD0 = np.mean(pDD   - dBY*zStepPX)
    
    f.write('reference PC (in pixels, PCX from left,  PCY from bottom):\n')
    f.write(str(pPCX0)+ '\n')
    f.write(str(pPCY0)+ '\n')
    f.write(str(pDD0)+ '\n')
    
    f.write('reference PC in PCX,PCY,DD ,  PCY from top/Height, DD =/Height):\n')
    f.write(str(pPCX0/ImageWidthPX)+ '    # measured from LEFT of pattern, in units of PatternWIDTH\n')
    f.write(str(1.0-pPCY0/ImageHeightPX)+ '    # measured from TOP of pattern, in units of PatternHEIGHT \n')
    f.write(str(pDD0 /ImageHeightPX)+ '    # to sample direction, in units of PatternHEIGHT  \n')
    
    f.close() 
    
    # explicit maps with the PC values
    map_width =mapinfo['nwidth']
    map_height=mapinfo['nheight']
    
    # note iz is useless here, dd changes with iy
    ix,iy,_ = make_map_indices(map_width,map_height)
    
    pcx_px = pPCX0 + ix * (-hStepPX)
    pcy_px = pPCY0 + iy * (+vStepPX)
    dd_px  = pDD0  + iy * (+zStepPX)
    
    pcx_brkr =      pcx_px/ImageWidthPX
    pcy_brkr = (1.0-pcy_px/ImageHeightPX)
    dd_brkr  =      dd_px/ImageHeightPX
    
    np.savetxt(bcf_filename+'_PCX.map',pcx_brkr.reshape((map_height,map_width)), fmt='%10.6f')
    np.savetxt(bcf_filename+'_PCY.map',pcy_brkr.reshape((map_height,map_width)), fmt='%10.6f')
    np.savetxt(bcf_filename+'_DD.map' , dd_brkr.reshape((map_height,map_width)), fmt='%10.6f')
    
    return

    
def calibratePC_BRKR(pcdata,mapinfo,XTilt=-20.0):
    """
    This function calibrates the Projection Center from the current PC fit values
    in ScanPointList assuming that the step size and the sample tilt is known.
    This makes it possible to extrapolate all experimentally determined PC values
    to a common reference point (0,0) assuming a regular x-y grid of steps tilted by 
    around the detector X-axis.
    The extrapolated reference PC values are then averaged to result
    in the PC estimation.
    
       
    Assumptions: 
    ------------
    
    * the SEM beam X-scan is exactly parallel to the X-Tilt detector axis
    * Bruker: XTilt=(SampleTilt-90)-DetectorTilt (deg)
    * i.e. SampleTilt=70 usually leads to NEGATIVE XTilts!!!
    * the method produces PC values in a tilted rectangular region
    * no trapezoidal/projective distortion is considered
        
    """

    print('SCALING TRANSFORMATION PC Calibration...')
    print('Assuming pure XTilt, no trapez:', XTilt)

    f = open('XCDSCALIBPC_LOG.TXT', "w")
       
    f.write('SEM SCAN BASED Pattern Center calibration\n')
    
    step_size =  0.094     #mapinfo['hstep']
    ImageWidthPX  = 160
    ImageHeightPX = 120

    f.write(str(step_size)  + '    # step size (microns)\n')
    f.write(str(ImageWidthPX)  + '    # PatternWIDTH\n')
    f.write(str(ImageHeightPX) + '    # PatternHEIGHT\n')
    alpha=-XTilt*np.pi/180.0 
    f.write(str(XTilt) + '    # total tilt angle between sample surface normal and direction to PC\n')
    #relative step size
    hStepRel=StepSize/ebsdconst.BRKR_WIDTH_MICRONS 
    
    # how many pixels moves the pattern center per scan point
    hStepPX= hStepRel*ImageWidthPX
    vStepPX= hStepPX*np.cos(alpha) # just tilted, no dependence of hStep on vStep
    zStepPX= hStepPX*np.sin(alpha)
    
    f.write('assumed PC beam steps in PX:\n')
    f.write(str(-hStepPX)+ '    # horizontal PC step size in pixels\n')
    f.write(str(vStepPX)+ '    # vertical   PC step size in pixels\n')
    f.write(str(zStepPX)+ '    # z          PC step size in pixels\n')
    
    #meanBX=np.round(np.mean(ScanPointList[:,15]))
    #meanBY=np.round(np.mean(ScanPointList[:,16]))
    meanBX=0.0
    meanBY=0.0
    f.write('Reference beam indices:\n')    
    f.write(str(meanBX)+ '\n')
    f.write(str(meanBY)+ '\n')
    dBX=ScanPointList[:,12]-meanBX
    dBY=ScanPointList[:,13]-meanBY
    
    # make absolute values in pixels
    # get away from 2 different units for PCX,PCY,DD given in Bruker software
    pPCX=      ScanPointList[:, 9]  * ImageWidthPX
    pPCY= (1.0-ScanPointList[:,10]) * ImageHeightPX
    pDD =      ScanPointList[:,11]  * ImageHeightPX
    
    # back-interpolate to mean reference PC
    pPCX0= np.mean(pPCX  + dBX*hStepPX)
    pPCY0= np.mean(pPCY  - dBY*vStepPX)
    pDD0 = np.mean(pDD   - dBY*zStepPX)
    
    f.write('reference PC (in pixels, PCX from left,  PCY from bottom):\n')
    f.write(str(pPCX0)+ '\n')
    f.write(str(pPCY0)+ '\n')
    f.write(str(pDD0)+ '\n')
    
    f.write('reference PC in PCX,PCY,DD ,  PCY from top/Height, DD =/Height):\n')
    f.write(str(pPCX0/ImageWidthPX)+ '    # measured from LEFT of pattern, in units of PatternWIDTH\n')
    f.write(str(1.0-pPCY0/ImageHeightPX)+ '    # measured from TOP of pattern, in units of PatternHEIGHT \n')
    f.write(str(pDD0 /ImageHeightPX)+ '    # to sample direction, in units of PatternHEIGHT  \n')
    
    f.close() 
    
    # explicit maps with the PC values
    map_width =mapinfo['nwidth']
    map_height=mapinfo['nheight']
    
    # note iz is useless here, dd changes with iy
    ix,iy,_ = make_map_indices(map_width,map_height)
    
    pcx_px = pPCX0 + ix * (-hStepPX)
    pcy_px = pPCY0 + iy * (+vStepPX)
    dd_px  = pDD0  + iy * (+zStepPX)
    
    pcx_brkr =      pcx_px/ImageWidthPX
    pcy_brkr = (1.0-pcy_px/ImageHeightPX)
    dd_brkr  =      dd_px/ImageHeightPX
    
    np.savetxt(bcf_filename+'_PCX.map',pcx_brkr.reshape((map_height,map_width)), fmt='%10.6f')
    np.savetxt(bcf_filename+'_PCY.map',pcy_brkr.reshape((map_height,map_width)), fmt='%10.6f')
    np.savetxt(bcf_filename+'_DD.map' , dd_brkr.reshape((map_height,map_width)), fmt='%10.6f')
    
    return
    
    
    
    
def plotPC(secondary,fit,pc3=None,units='Bruker',plotdir=''):
    """
    plots the pattern center data and shows parameters
    """
    
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)    
    #plt.title(title,fontsize=26)
    plt.ylabel('PCY (pattern height from top)', color='k', fontsize=22)
    plt.gca().invert_yaxis() # consistent with y measured from top of pattern
    plt.xlabel('PCX (pattern width from left)', color='k',fontsize=22)
    plt.scatter(fit.T[0],fit.T[1],s=22,c='red',alpha=1.0, lw=0)
    plt.scatter(secondary.T[0],secondary.T[1],s=22,c='blue',alpha=1.0,lw=0)
    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(plotdir+'/PCXY_FIT.png',dpi=300,bbox_inches = 'tight')
    #plt.show()
    plt.close()
   
   
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)    
    #plt.title(title,fontsize=26)
    plt.ylabel('DD  (pattern height)', color='k', fontsize=22)
    plt.xlabel('PCX (pattern width from left)', color='k',fontsize=22)
   
    plt.scatter(fit.T[0],fit.T[2],s=22,c='red',alpha=1.0,lw=0)
    plt.scatter(secondary.T[0],secondary.T[2],s=22,c='blue',alpha=1.0,lw=0)
    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(plotdir+'/PCXZ_FIT.png',dpi=300,bbox_inches = 'tight')
    #plt.show()
    plt.close()
   
   
    # linear fit to YZ-curve to estimate tilt angle
    slope, intercept, r_value, p_value, std_err_slope = stats.linregress(fit.T[2],fit.T[1])    
    print(slope, intercept, r_value, p_value, std_err_slope)
    XTilt=-np.abs(np.arctan(1.0/slope)*180.0/np.pi)
    print('approximated total XTilt=(SampleTilt-90)-DetectorTilt (deg):',XTilt)
    fittedY=intercept+slope*fit.T[2]
    
    
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)    
    #plt.title(title,fontsize=26)
    plt.xlabel('DD (pattern height)', color='k', fontsize=22)
    plt.ylabel('PCY (pattern height from top)', color='k',fontsize=22)
    plt.gca().invert_yaxis() # consistent with y measured from top of pattern
    plt.scatter(fit.T[2],fit.T[1],s=22,c='red',alpha=1.0,lw=0)
    plt.scatter(secondary.T[2],secondary.T[1],s=22,c='blue',alpha=1.0,lw=0)
    plt.plot(fit.T[2],fittedY,'r-' )
    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(plotdir+'/PCYZ_FIT.png',dpi=300,bbox_inches = 'tight')
    #plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.gca().invert_yaxis() # consistent with y measured from top of pattern
    
    ax.scatter(secondary.T[0],secondary.T[2],secondary.T[1],s=16, c='b', marker='o',lw=0,alpha=1.0)
    ax.scatter(fit.T[0],fit.T[2],fit.T[1], c='red', s=16, marker='o',lw=0,alpha=1.0)
    
    plt.grid(True)
    ax.set_xlabel('PCX')
    ax.set_ylabel('DD')
    ax.set_zlabel('PCY')
    plt.savefig(plotdir+'/PCXYZ_FIT.png',dpi=300,bbox_inches = 'tight')
    
    #plt.show()
    plt.close()
    
    return XTilt

    
def normrange(Y):
    """normalize the range of the Y values
    """
    Ymax=np.amax(Y,axis=0)
    Ymin=np.amin(Y,axis=0)
    dY  =Ymax-Ymin
    Yn=(Y-Ymin)/dY - 0.5  
    return Yn
    

def make_map_indices(w,h):
    """
    makes 3  1D arrays of x and y, z=1 indices of a map w*h
    for projective transformations
    """
    ix = np.tile(np.arange(0, w), h)
    iy = np.reshape(np.tile(np.arange(0, h), [w,1]).T, -1)
    iz = np.ones_like(ix)
    return ix,iy,iz    
    
def project_points(pts_src_3d,h):
    """ make 2D plane-projected points from homography matrix
    """
    # to get same result with numpy
    fit = np.dot(pts_src_3d,h.T)
    projected = fit / fit[:,2,np.newaxis] # dehomogenize
    projected[:,2]=0.0
    #print(fit)
    
    #using OpenCV, assumes 2d points
    #pts_src=np.array([pts_src]) # 2d
    #projected = cv2.perspectiveTransform(pts_src, h)[0]
    return projected    
    

def make_projective_PCdata(A,MapWidth,MapHeight,outfile=None):
    """
    apply projective transformation matrix to all map indices
    
    TODO: 1. save projection center values for direct import into DynamicS
          2. save PC values in HDF5 
    """
    # list of all beam index coordinates
    # add z=1 for homogeneous coordinates in 2D
    ix,iy,iz=make_map_indices(MapWidth,MapHeight)
    
    map_coo=np.array([ix,iy,iz]).T
    
    # apply projective transformation
    #pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    #unpad = lambda x: x[:,:-1]
    #transform = lambda x: unpad(np.dot(pad(x), A))
    pc_coo=np.dot(map_coo,A)
    #pc_coo=transform(map_coo)
    
    if outfile:
        np.savetxt(outfile+'_A_proj.dat',A)
        np.savetxt(outfile+'_PCX.map',pc_coo.T[0].reshape((MapHeight,MapWidth)))
        np.savetxt(outfile+'_PCY.map',pc_coo.T[1].reshape((MapHeight,MapWidth)))
        np.savetxt(outfile+'_DD.map' ,pc_coo.T[2].reshape((MapHeight,MapWidth)))

    return
    
    
    
    

    

    
def make_interpolated_PCdata(xc_beam_indices,xc_pc_coords,nwidth,nheight,outfile=''):
    """
    interpolates between projection center values, input assumes column vectors
    
    Radial Basis Functions for interpolation of scattered data:
    http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    """

    print('Interpolating Projection Center Map Data...')
    
    ix=xc_beam_indices[0]
    iy=xc_beam_indices[1]
    
    f_pcx = Rbf(ix, iy, xc_pc_coords[0,:])    
    f_pcy = Rbf(ix, iy, xc_pc_coords[1,:]) 
    f_dd  = Rbf(ix, iy, xc_pc_coords[2,:]) 
    
    xind = np.arange(0,nwidth)
    yind = np.arange(0,nheight)
    ix_map, iy_map = np.meshgrid(xind, yind)
    
    pcx= f_pcx(ix_map, iy_map)
    pcy= f_pcy(ix_map, iy_map)
    dd =  f_dd(ix_map, iy_map)
    
    if outfile>'':
        np.savetxt(outfile+'_PCX_interp.map',pcx, fmt='%10.6f')
        np.savetxt(outfile+'_PCY_interp.map',pcy, fmt='%10.6f')
        np.savetxt(outfile+'_DD_interp.map' ,dd , fmt='%10.6f')
        
    return


     
    
def PCstats(DataMatrix0,mean_dim,title):
    """ Statistics of pattern center positions
    
    PCX, PCY, and  DD should be constant along rows or colums respectively
    thus the stdDev along a column gives an error estimate 
    """
    if (mean_dim==1):
        DataMatrix=DataMatrix0.T
    else:
        DataMatrix=DataMatrix0
        
    # mask all zero values (skipped map points)
    #DataMatrix=np.ma.masked_values(DataMatrix0,0.0)
    print(DataMatrix)
    print(title)    
    print('Mean in best fit:');
    # make data sets for row/column means and std

    PCMean=np.zeros(DataMatrix.shape[0])
    PCStd=np.zeros(DataMatrix.shape[0])
    for i in range(DataMatrix.shape[0]):
        PCMean[i]=np.mean(DataMatrix[i,1:-1]) # exclude first and last point
        PCStd[i]=np.std(DataMatrix[i,1:-1]) # exclude first and last point
        if not (np.isnan(PCMean[i]) or np.isnan(PCStd[i])):
            print(i+1, PCMean[i], PCStd[i]) 

    print('Mean error bar (all valid points):',np.mean(np.ma.masked_invalid(PCStd)[1:-1]))    

    PCMean=np.ma.masked_invalid(PCMean)
    BeamPos=np.ma.masked_where(np.ma.getmask(PCMean), np.arange(DataMatrix.shape[0]))
    #print(BeamPos,PCMeanMasked)
    #print(len(BeamPos),len(PCMeanMasked))
    mask = np.isfinite(PCMean) & np.isfinite(BeamPos)
    slope, intercept, r_value, p_value, std_err_slope = stats.linregress(BeamPos[mask][1:-1],PCMean[mask][1:-1])    
    print(slope, intercept, r_value, p_value, std_err_slope)

    fittedY=intercept+slope*BeamPos
    residuals_y=PCMean-fittedY
    print('StdDev of Y residuals:',np.std(residuals_y))
    #print(fittedY)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)    
    plt.title(title,fontsize=26)
    plt.ylabel('position (microns)', color='k', fontsize=22)
    plt.xlabel('beam scan index', color='k',fontsize=22)
    plt.plot(BeamPos[mask],fittedY[mask],'r-' )
    plt.errorbar(BeamPos,PCMean,yerr=PCStd,fmt='o')
    plt.savefig(title+'.png',dpi=300,bbox_inches = 'tight')
    plt.show()
    plt.close()
    print('')    