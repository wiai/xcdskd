import time
import sys

import numpy as np

from . import kikufilter
from ..io.progress import print_progress_line

def asy(data1,data2):
    """
    difference of the two data sets, normalized by the sum of both
    """
    return (data1-data2)/(data1+data2)
    
    
def make2Dmap(Data1D,Xidx,Yidx,NHeight,NWidth):
    ''' make 2D map array from 1D Data1D list with index values in Xidx and Yidx
    final map array height and width are NHeight,NWidth
    '''
    #resulting 2D data set
    Map2D=np.zeros((NHeight,NWidth),dtype=Data1D.dtype)
    # fill map with values from 1D data
    Map2D[Yidx,Xidx]=Data1D.T
    return Map2D  


def normalizeChannel(data,AddBright=0,Contrast=1.0):
    """ normalize a numpy array to mean=0 and stddev=1.0 
        and then scale by contrast factor and shift brightness values 
    """
    value=np.array((128+AddBright)+Contrast*80*(data-np.median(data))/np.std(data))
    return np.clip(value,0,255)

    
def get_vrange(signal, stretch=2.0):
    med=np.median(signal)
    std=np.std(signal)
    vmin=med-stretch*std
    vmax=med+stretch*std
    return [vmin,vmax]    
    
    

def TotalSum(PatternMatrix):
    return np.sum(PatternMatrix)

def getBGfac(kiku,background):
    """
    estimate scaling factor for FFT background
    in order to have always positive Kikuchi signal
    (the signal that remains after subtracting the background from the raw pattern)
    """
    offset=-np.min(kiku)
    #print(offset)
    MinXY = np.unravel_index(kiku.argmin(), kiku.shape)
    #print MinXY
    #print KIK[MinXY[0],MinXY[1]]
    RefValue=background[MinXY[0],MinXY[1]]
    #print RefValue
    bgfac = (RefValue-offset)/RefValue
    return bgfac


    
def CalcFSDROIs(Patterns,SignalSelection,MH,MW,XIndex,YIndex,bgscale=1.0,vfsd_frac=0.1,flip=False):
    """
    calc virtual FSD channels from EBSD patterns
    """
    KKMapMin=100000000.0
    BGfac=1000.0
    MaxNegativeIntensity=10000.0
    KKTotalNegError=0.0
    
    if (SignalSelection=='KIK') or  (SignalSelection=='BG'): 
        print('FFT background scaling factor: '+str(bgscale))
    ScalingFac=1000.0
    FSD1=np.zeros(Patterns.shape[0])
    FSD2=np.zeros(Patterns.shape[0])
    FSD3=np.zeros(Patterns.shape[0])
    FSD4=np.zeros(Patterns.shape[0])
    FSD5=np.zeros(Patterns.shape[0])
    FSD6=np.zeros(Patterns.shape[0])

    FullInt=np.zeros(Patterns.shape[0])
    #TopInt=np.zeros(Patterns.shape[0])
    pattern_width=Patterns.shape[2] # width
    pattern_third=pattern_width // 3 
    pattern_height=Patterns.shape[1] #height
    
    TopBottomLines=int(vfsd_frac*pattern_height)
    print('No. of Top / Bottom Lines for FSDs: '+str(TopBottomLines));
    
    tstart = time.time()
    for i in range(Patterns.shape[0]):
        # get current pattern
        #RawPattern=np.flipud(Patterns[i,:,:].T)
        RawPattern=Patterns[i,:,:]
        if flip:
            RawPattern=np.flipud(RawPattern)
        
        if (SignalSelection=='KIK') or  (SignalSelection=='BG'): 
            # subtract the background
            KK,FFTBackGround=kikufilter.BGK(RawPattern,15,support=(50,50),bgscale=0.9)
            KKsum=np.sum(KK)
        
            # check physicality of Background
            KKneg=KK[KK<0]
            negIntRatio=np.sum(KKneg)/KKsum
            KKTotalNegError=KKTotalNegError+negIntRatio
            if negIntRatio<MaxNegativeIntensity:
                MaxNegativeIntensity=negIntRatio
            
        if SignalSelection=='KIK': 
            Kiku=KK
            if Kiku.min()<KKMapMin:
                KKMapMin=Kiku.min()
            bgf=getBGfac(KK,FFTBackGround)
            if bgf<BGfac:
                BGfac=bgf
        if SignalSelection=='BG': 
            Kiku=FFTBackGround
            #Kiku=Background
        if SignalSelection=='RAW': 
            Kiku=RawPattern
        
        KikuROI=Kiku[pattern_height-TopBottomLines:pattern_height, 0:pattern_third]
        FSD1[i]=TotalSum(KikuROI)/ScalingFac
        
        KikuROI=Kiku[pattern_height-TopBottomLines:pattern_height, pattern_third:2*pattern_third]
        FSD2[i]=TotalSum(KikuROI)/ScalingFac 

        KikuROI=Kiku[pattern_height-TopBottomLines:pattern_height, 2*pattern_third:pattern_width]
        FSD3[i]=TotalSum(KikuROI)/ScalingFac
        
        # top ROIs
        KikuROI=Kiku[0:TopBottomLines, 0:pattern_third]
        FSD4[i]=TotalSum(KikuROI)/ScalingFac
        
        KikuROI=Kiku[0:TopBottomLines, pattern_third:2*pattern_third]
        FSD5[i]=TotalSum(KikuROI)/ScalingFac 

        KikuROI=Kiku[0:TopBottomLines, 2*pattern_third:pattern_width]
        FSD6[i]=TotalSum(KikuROI)/ScalingFac
        
        FullInt[i]=TotalSum(Kiku)/ScalingFac
        
        # update time info every 100 patterns
        if (i % 100 == 0):
            progress=100.0*(i+1)/Patterns.shape[0]
            tup = time.time()
            togo = (100.0-progress)*(tup-tstart)/(60.0*progress)
            if (SignalSelection=='KIK') or  (SignalSelection=='BG'): 
                sys.stdout.write("\rtotal map points:%5i current:%5i progress: %4.2f%% -> %6.1f min to go  ave. Kiku intensity error: %6.4f%%" % (Patterns.shape[0],i+1,progress,togo,100.0*KKTotalNegError/(i+1))  )
            else:
                sys.stdout.write("\rtotal map points:%5i current:%5i progress: %4.2f%% -> %6.1f min to go" % (Patterns.shape[0],i+1,progress,togo))

            sys.stdout.flush()
    
    if SignalSelection=='KIK': 
        print('\n Global Map Minimum of Kikuchi Signal:'+str(KKMapMin))
        print('Global Minimum Background Scaling Factor:'+str(BGfac))
    
    # make 2D Arrays
    left0     =make2Dmap(FSD1,XIndex,YIndex,MH,MW)
    middle0   =make2Dmap(FSD2,XIndex,YIndex,MH,MW)
    right0    =make2Dmap(FSD3,XIndex,YIndex,MH,MW)
    topleft0  =make2Dmap(FSD4,XIndex,YIndex,MH,MW)
    topmiddle0=make2Dmap(FSD5,XIndex,YIndex,MH,MW)
    topright0 =make2Dmap(FSD6,XIndex,YIndex,MH,MW)
    full0     =make2Dmap(FullInt,XIndex,YIndex,MH,MW)
    
    # mask the measured area
    full  = np.ma.masked_less_equal(full0,0.0001)
    left  = np.ma.array(left0  ,mask=full.mask)
    right = np.ma.array(right0 ,mask=full.mask)
    middle= np.ma.array(middle0,mask=full.mask)
    topleft  = np.ma.array(topleft0  ,mask=full.mask)
    topright = np.ma.array(topright0 ,mask=full.mask)
    topmiddle= np.ma.array(topmiddle0,mask=full.mask)
    
    
    # left, middle, right, full, top signals
    return [full, left, middle, right, topleft, topmiddle, topright]
    
    
def CalcMaskFSDs(mask_a,mask_b,SignalSelection='RAW',bgscale=1.0):
    """
    calc virtual FSD channels from mask region
    """
    FullInt=np.zeros(Patterns.shape[0])
    a_int=np.zeros(Patterns.shape[0])
    b_int=np.zeros(Patterns.shape[0])
    asy=np.zeros(Patterns.shape[0])
    
    KKMapMin=100000000.0
    BGfac=1000.0
    MaxNegativeIntensity=10000.0
    KKTotalNegError=0.0
    ScalingFac=1000.0
    
    tstart = time.time()
    for i in range(Patterns.shape[0]):
        # get current pattern
        RawPattern=Patterns[i,:,:]
        
        if (SignalSelection=='KIK') or  (SignalSelection=='BG'): 
            # subtract the background
            KK,FFTBackGround=kikufilter.BGK(RawPattern,15,support=(50,50),bgscale=0.9)
            KK=RawPattern-FFTBackGround
            KKsum=np.sum(KK)
        
            # check physicality of Background
            KKneg=KK[KK<0]
            negIntRatio=np.sum(KKneg)/KKsum
            KKTotalNegError=KKTotalNegError+negIntRatio
            if negIntRatio<MaxNegativeIntensity:
                MaxNegativeIntensity=negIntRatio
            
        if SignalSelection=='KIK': 
            Kiku=KK
            if Kiku.min()<KKMapMin:
                KKMapMin=Kiku.min()
            bgf=getBGfac(KK,FFTBackGround)
            if bgf<BGfac:
                BGfac=bgf
                
        if SignalSelection=='RAW': 
            Kiku=RawPattern
        
        kiku2=np.copy(Kiku)
        kiku1=np.copy(Kiku)
        
        # apply the mask to the un/processed patterns
        kiku_mask_a=np.ma.masked_less(mask_a,1)
        kiku_mask_b=np.ma.masked_less(mask_b,1)
   
        kiku_a  = np.ma.array(kiku1 ,mask=kiku_mask_a.mask)
        kiku_b  = np.ma.array(kiku2 ,mask=kiku_mask_b.mask)
    
        a_int[i]=np.mean(kiku_a)
        b_int[i]=np.mean(kiku_b)
        #print(a_int[i],b_int[i])
        
        asy=(a_int-b_int)/(a_int+b_int)
        FullInt[i]=TotalSum(RawPattern)/ScalingFac
        
        # update time info every 100 patterns
        if (i % 100 == 0):
            progress=100.0*(i+1)/Patterns.shape[0]
            tup = time.time()
            togo = (100.0-progress)*(tup-tstart)/(60.0*progress)
            sys.stdout.write("\rtotal map points:%5i current:%5i progress: %4.2f%% -> %6.1f min to go  ave. Kiku intensity error: %6.4f%%" % (Patterns.shape[0],i+1,progress,togo,100.0*KKTotalNegError/(i+1))  )
            sys.stdout.flush()
    
   
    # make 2D Arrays
    full0  = make2Dmap(FullInt,XIndex,YIndex,MapHeight,MapWidth)
    asymm0 = make2Dmap(asy    ,XIndex,YIndex,MapHeight,MapWidth)
 
    # mask the measured area
    full  = np.ma.masked_less_equal(full0,0.0001)
    asymm = np.ma.array(asymm0  ,mask=full.mask)
    
    return full, asymm




def calc_COI_raw(Patterns,progress=True,flip=False):
    """
    calculate center of mass of raw image intensity = COI 
    """
    NumberOfPatterns=Patterns.shape[0]
    COIxp=np.zeros(NumberOfPatterns)
    COIyp=np.zeros(NumberOfPatterns)
    tstart = time.time()
    for i in range(Patterns.shape[0]):
        if flip:
            COIxp[i],COIyp[i]=improperties.getCOIpix(np.flipud(Patterns[i]))
        else:
            COIxp[i],COIyp[i]=improperties.getCOIpix(Patterns[i])
            
        # update time info every 100 patterns
        if (i % 100 == 0) and progress:
            progress=100.0*(i+1)/Patterns.shape[0]
            tup = time.time()
            togo = (100.0-progress)*(tup-tstart)/(60.0*progress)
            sys.stdout.write("\rtotal map points:%5i current:%5i progress: %4.2f%% -> %6.1f min to go" % (Patterns.shape[0],i+1,progress,togo)  )
            sys.stdout.flush()        
        
    return COIxp,COIyp     

def getCOIpix(pattern):
    """ 
    returns x,y-center of intensity pixel coordinates 
    of a pattern matrix
    
    coordinates parallel to DETECTOR SYSTEM
    """
    pattern = pattern.astype(np.float64)
    
    nx = pattern.shape[1] # x = columns from left
    ny = pattern.shape[0] # y = rows from top
   
    # weight intensities by pixel number
    xi = pattern * np.arange(0, nx, dtype=np.int16)[:,np.newaxis].T
    yi = pattern * np.arange(0, ny, dtype=np.int16)[:,np.newaxis]
    
    # weighted sums
    PatternSum = np.sum(pattern)
    xp_coi = np.sum(xi)/PatternSum
    yp_coi = ny-np.sum(yi)/PatternSum
    
    return xp_coi,yp_coi
    
    
def getstdpix(pattern,mean_x,mean_y):
    """ 
    returns 2nd moment of intensity distribution 
    coordinates parallel to DETECTOR SYSTEM
    see e.g. Bulmer, Principles of Statistics p.57
    """
    nx=pattern.shape[1] # x = columns from left
    ny=pattern.shape[0] # y = rows from top
    #print('x (hor), y (ver) pixels: ',nx,ny)
   
    # pixel coordinates
    xp=np.arange(0,nx,dtype=int)[:,np.newaxis].T
    yp=np.arange(0,ny,dtype=int)[:,np.newaxis]
   
   
    # weight intensities by pixel number
    #xi=pattern  * np.arange(0,nx,dtype=float)[:,np.newaxis].T
    #yi=pattern  * np.arange(0,ny,dtype=float)[:,np.newaxis]
    
    # weighted sums
    PatternSum=np.sum(pattern)
    xp_std=np.sum(pattern*(xp-mean_x)**2)/(nx-1)
    yp_std=np.sum(pattern*(yp-mean_y)**2)/(ny-1)
    
    return xp_std,yp_std    
 


 
def CalcCOIpix_RawKik(Patterns):
    """
    calc center of mass of image intensity
    """
    NumberOfPatterns=Patterns.shape[0]
    COMxp=np.zeros((NumberOfPatterns,4))
    COMyp=np.zeros((NumberOfPatterns,4))
    tstart = time.time()
    sigma=np.max( [Patterns.shape[1]//20,15])
    print('CalcCOIpix sigma for background: ',sigma)
    
    for i in range(Patterns.shape[0]):
        # get current pattern
        RawPattern=Patterns[i,:,:]
        KK,FFTBackGround=kikufilter.BGK(RawPattern,sigma,support=(3*sigma,3*sigma),bgscale=0.95)
        COMxp[i,0],COMyp[i,0]=getCOIpix(RawPattern)
        COMxp[i,1],COMyp[i,1]=getCOIpix(FFTBackGround)
        COMxp[i,2],COMyp[i,2]=getCOIpix(KK)
        COMxp[i,3],COMyp[i,3]=getCOIpix(KK/FFTBackGround)
                
        print_progress_line(tstart, i, npatterns, 100)        
        
    return COMxp,COMyp        
    
    
def calc_COM_px(patterns, process=None):
    """
    calc center of mass of image intensity,
    "process" is an optional pre-processing function
    """
    npatterns=patterns.shape[0]
    COMxp=np.zeros((npatterns))
    COMyp=np.zeros((npatterns))
    tstart = time.time()
    for i in range(npatterns):
        # get current pattern
        if process is None:
            COMxp[i],COMyp[i]=getCOIpix(patterns[i,:,:])
        else:
            COMxp[i],COMyp[i]=getCOIpix(process(patterns[i,:,:]))

        # update time info every 100 patterns
        print_progress_line(tstart, i, npatterns, 100)      

    return COMxp,COMyp  
    
    
    
def remove_vgradient(src_image, window=0):
    """ subtracts top-bottom gradient from image 
    """
    width=src_image.shape[1]
    dst_image = np.zeros_like(src_image)
    total_mean=np.nanmean(src_image)    
    hmax=src_image.shape[0]-1
    
    col_start=int(0.05*width)
    col_end  =int(0.95*width)

    for iline in range(hmax+1):
        window_med = 0.0
        for iwindow in range(-window, window+1):
            il = iline + iwindow
            if (il > hmax):
                il = hmax - (il - hmax) 
            if (il < 0):    
                il = -il
            window_med += np.median(src_image[il, col_start:col_end])

        window_med = window_med / (2 * window + 1)
        dst_image[iline,:] = src_image[iline,:] - window_med
    #line_means=np.mean(src_image[:,col_start:col_end], axis=1)
    #total_mean=np.mean(line_means)
    #dst_image=total_mean+(src_image-line_means.T[:,np.newaxis])    
    return dst_image


    
def remove_hgradient(src_image, **kwargs):
    """ subtracts left-right gradient from image
        replaces mean value of src_image for mean of result 
    """
    dst_image=remove_vgradient(np.asarray(src_image).T, **kwargs).T
    return dst_image
    
    
def remove_hvgradient(src_image):
    """ subtracts left-right and top-bootm gradient from image
        replaces mean value of src_image for mean of result 
    """
    dst_image=remove_vgradient(remove_hgradient(src_image))
    return dst_image       


def flatten(image, method="plane", norm=False):
    """ flatten image by plane subtraction
    """
    im = spiepy.Im()
    im.data = np.array(image)
    if method=="poly2":
        im_out, im_bg = spiepy.flatten_poly_xy(im) 
    if method=="plane":
        im_out, im_bg = spiepy.flatten_xy(im)    
        
    if norm:
        out_data= im_out.data/im_bg
    else:
        out_data= im_out.data     
      
    return out_data    
    
    
    
def calc_mean_direction(unit_vectors):
    """ calculate the mean vector of unit vectors on the sphere
    
    Sources:

    Asymptotic Behavior of Sample Mean Direction for Spheres 
    Harrie Hendriks, Zinoviy Landsman, Frits Ruymgaart
    Journal of Multivariate Analysis 59, 1996, Pages 141-152
    http://www.sciencedirect.com/science/article/pii/S0047259X96900573
    
    Mardia, K. V. (1975). Statistics of Directional Data. Academic Press, London.
    Watson, G. S. (1983). Statistics on Spheres. Wiley, New York.
    """
    VSum=np.sum(unit_vectors,axis=0)
    SSum=np.sum(VSum**2     ,axis=0)
    R = np.sqrt(SSum)
    Vm = VSum/R
    return Vm  



def rebin_array(a, new_rows=7, new_cols=7): 
    '''
    This function takes an 2D numpy array a and produces a smaller array 
    of size new_rows, new_cols. new_rows and new_cols must be less than 
    or equal to the number of rows and columns in a.
    
    https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    
    '''
    rows = a.shape[0] #len(a)
    cols = a.shape[1] #len(a[0])
    yscale = float(rows) / new_rows 
    xscale = float(cols) / new_cols

    # first average across the cols to shorten rows    
    new_a = np.zeros((rows, new_cols)) 
    for j in range(new_cols):
        # get the indices of the original array we are going to average across
        the_x_range = (j*xscale, (j+1)*xscale)
        firstx = int(the_x_range[0])
        lastx = int(the_x_range[1])
        # figure out the portion of the first and last index that overlap
        # with the new index, and thus the portion of those cells that 
        # we need to include in our average
        x0_scale = 1 - (the_x_range[0]-int(the_x_range[0]))
        xEnd_scale =  (the_x_range[1]-int(the_x_range[1]))
        # scale_line is a 1d array that corresponds to the portion of each old
        # index in the_x_range that should be included in the new average
        scale_line = np.ones((lastx-firstx+1))
        scale_line[0] = x0_scale
        scale_line[-1] = xEnd_scale
        # Make sure you don't screw up and include an index that is too large
        # for the array. This isn't great, as there could be some floating
        # point errors that mess up this comparison.
        if scale_line[-1] == 0:
            scale_line = scale_line[:-1]
            lastx = lastx - 1
        # Now it's linear algebra time. Take the dot product of a slice of
        # the original array and the scale_line
        new_a[:,j] = np.dot(a[:,firstx:lastx+1], scale_line)/scale_line.sum()

    # Then average across the rows to shorten the cols. Same method as above.
    # It is probably possible to simplify this code, as this is more or less
    # the same procedure as the block of code above, but transposed.
    # Here I'm reusing the variable a. Sorry if that's confusing.
    a = np.zeros((new_rows, new_cols))
    for i in range(new_rows):
        the_y_range = (i*yscale, (i+1)*yscale)
        firsty = int(the_y_range[0])
        lasty = int(the_y_range[1])
        y0_scale = 1 - (the_y_range[0]-int(the_y_range[0]))
        yEnd_scale =  (the_y_range[1]-int(the_y_range[1]))
        scale_line = np.ones((lasty-firsty+1))
        scale_line[0] = y0_scale
        scale_line[-1] = yEnd_scale
        if scale_line[-1] == 0:
            scale_line = scale_line[:-1]
            lasty = lasty - 1
        a[i:,] = np.dot(scale_line, new_a[firsty:lasty+1,])/scale_line.sum() 

    return a 



def make_vbse_array(patterns, nrows=7, ncols=7, process=None):
    """ calculate n x n average regions of pattern """
    npatterns = patterns.shape[0]
    vfsd = np.zeros((npatterns, nrows , ncols))
    tstart = time.time()
    for i in range(npatterns):
        if process is None:
            vfsd[i] = rebin_array(patterns[i,:,:], nrows, ncols)
        else:
            vfsd[i] = rebin_array(process(patterns[i,:,:]), nrows, ncols)
        # update progress info
        print_progress_line(tstart, i, npatterns, 100)
      
    return vfsd  




def light_direction(azdeg=315, altdeg=45):
    """ The unit vector direction towards the light source """

    # Azimuth is in degrees clockwise from North. Convert to radians
    # counterclockwise from East (mathematical notation).
    az = np.radians(90 - azdeg)
    alt = np.radians(altdeg)

    return np.array([
        np.cos(az) * np.cos(alt),
        np.sin(az) * np.cos(alt),
        np.sin(alt)
    ])


def _vector_magnitude(arr):
    # things that don't work here:
    #  * np.linalg.norm
    #    - doesn't broadcast in numpy 1.7
    #    - drops the mask from ma.array
    #  * using keepdims - broken on ma.array until 1.11.2
    #  * using sum - discards mask on ma.array unless entire vector is masked

    sum_sq = 0
    for i in range(arr.shape[-1]):
        sum_sq += np.square(arr[..., i, np.newaxis])
    return np.sqrt(sum_sq)


def _vector_dot(a, b):
    # things that don't work here:
    #   * a.dot(b) - fails on masked arrays until 1.10
    #   * np.ma.dot(a, b) - doesn't mask enough things
    #   * np.ma.dot(a, b, strict=True) - returns a maskedarray with no mask
    dot = 0
    for i in range(a.shape[-1]):
        dot += a[..., i] * b[..., i]
    return dot


def shade_normals(normals, fraction=1., azdeg=315, altdeg=45):
    """
    Calculates the illumination intensity for the normal vectors of a
    surface using the defined azimuth and elevation for the light source.
    Imagine an artificial sun placed at infinity in some azimuth and
    elevation position illuminating our surface. The parts of the surface
    that slope toward the sun should brighten while those sides facing away
    should become darker.
    
    Parameters
    ----------
    fraction : number, optional
        Increases or decreases the contrast of the hillshade.  Values
        greater than one will cause intermediate values to move closer to
        full illumination or shadow (and clipping any values that move
        beyond 0 or 1). Note that this is not visually or mathematically
        the same as vertical exaggeration.
        
    Returns
    -------
    intensity : ndarray
        A 2d array of illumination values between 0-1, where 0 is
        completely in shadow and 1 is completely illuminated.
    """

    direction = light_direction(azdeg=azdeg, altdeg=altdeg)
    intensity = _vector_dot(normals, direction)

    # Apply contrast stretch
    imin, imax = intensity.min(), intensity.max()
    intensity *= fraction

    # Rescale to 0-1, keeping range before contrast stretch
    # If constant slope, keep relative scaling (i.e. flat should be 0.5,
    # fully occluded 0, etc.)
    if (imax - imin) > 1e-6:
        # Strictly speaking, this is incorrect. Negative values should be
        # clipped to 0 because they're fully occluded. However, rescaling
        # in this manner is consistent with the previous implementation and
        # visually appears better than a "hard" clip.
        intensity -= imin
        intensity /= (imax - imin)
    intensity = np.clip(intensity, 0, 1, intensity)

    return intensity




def calc_shading_map(comx, comy, fraction=1.0, azdeg=-60, altdeg=20):
    """
    normal shading for 2D maps of comx, comy
    """
    normals = np.empty(comx.shape + (3,))
    normals[..., 0] = comx
    normals[..., 1] = comy
    normals[..., 2] = 1
    normals /= _vector_magnitude(normals)
    shading=shade_normals(normals, fraction=fraction, azdeg=azdeg, altdeg=altdeg)
    return shading[..., np.newaxis]
