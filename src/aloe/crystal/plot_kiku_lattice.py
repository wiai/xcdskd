#------------------------------------------------------------------------------
# Plotting of lattice plane traces
# for EBSD patterns in gnomonic projection
#------------------------------------------------------------------------------
#
# Limitations:
# only crystal-geometric calculations: LATTICE PLANE TRACES and ZONE AXES 
# no atoms, no extinction rules, no Kikuchi bands!
#
#

#----------------- PLOTTING OPTIONS -------------------------------------------
# use pattern image 
PLOT_PATTERN_IMAGE=True
# zoom factor for space around EBSD-pattern:
PATTERN_PLOT_FACTOR=1.2
# plot (hkl) plane poles:  
PLOT_HKL_POLES =False
# draw projection of reciprocal space direction:
PLOT_HESSE_LINES=False
# draw gnomonic angles in 10 degree intervals:
PLOT_GNOM_THETA=True
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# USER PARAMETERS: ENTER the Values from the BRUKER ESPRIT & DYNAMICS EBSD SOFTWARE
#------------------------------------------------------------------------------

ExpImageFilename='../data/Ni_Example3.bmp'

a =1
b =1
c =1
alpha = 90.0 # convert angles into radians
beta  = 90.0
gamma = 90.0

BRKR_SampleTilt=70.0
BRKR_DetectorTilt=4.58

# Euler Angles (Bunge convention, rotation of the coordinate system base vectors)
BRKR_phi1=126.7
BRKR_Phi=37.9
BRKR_phi2=272.2

# Pattern Center
BRKR_DD =0.642 # in units of screen height
BRKR_PCX=0.483 # in units of screen width, from left of exp. pattern
BRKR_PCY=0.279 # in units of screen height, from top of exp. pattern

# Screen, only the Aspect Ratio is important here
BRKR_ScreenWidth=320
BRKR_ScreenHeight=230

# generate (hkl) up to (+/-hkl_max +/-hkl_max +/-hkl_max)
hkl_max=1
h_step=1
k_step=1
l_step=1


ExpImageFilename='../data/Ni_Example3.bmp'

a =1
b =1
c =1
alpha = 90.0 # convert angles into radians
beta  = 90.0
gamma = 90.0

BRKR_SampleTilt=70.0
BRKR_DetectorTilt=4.58

# Euler Angles (Bunge convention, rotation of the coordinate system base vectors)
BRKR_phi1=126.7
BRKR_Phi=37.9
BRKR_phi2=272.2

# Pattern Center
BRKR_DD =0.642 # in units of screen height
BRKR_PCX=0.483 # in units of screen width, from left of exp. pattern
BRKR_PCY=0.279 # in units of screen height, from top of exp. pattern

# Screen, only the Aspect Ratio is important here
BRKR_ScreenWidth=320
BRKR_ScreenHeight=230

# generate (hkl) up to (+/-hkl_max +/-hkl_max +/-hkl_max)
hkl_max=1
h_step=1
k_step=1
l_step=1




#------------------------------------------------------------------------------
# triclinic reference test values
#------------------------------------------------------------------------------


'''
ExpImageFilename='triclinic_22_39_67_085_035_015_70_-7.png'

a =2.0
b =3.0
c =4.0
alpha = 70.0
beta  = 100.0
gamma = 120.0

BRKR_SampleTilt=70.0
BRKR_DetectorTilt=-7.0

# Euler Angles (Bunge convention, rotation of the coordinate system base vectors)
BRKR_phi1=22
BRKR_Phi=39
BRKR_phi2=67

# Pattern Center
BRKR_DD =0.85 # in units of screen height
BRKR_PCX=0.35 # in units of screen width, from left of exp. pattern
BRKR_PCY=0.15 # in units of screen height, from top of exp. pattern

# Screen, only the Aspect Ratio is important here
BRKR_ScreenWidth=320
BRKR_ScreenHeight=230

# generate (hkl) up to (+/-hkl_max +/-hkl_max +/-hkl_max)
hkl_max=1
h_step=1
k_step=1
l_step=1
#------------------------------------------------------------------------------
'''


'''
# orthoclase (monoclinic) example
ExpImageFilename='orthoclase.png'

a =7.21
b =8.55
c =12.97
alpha = 90.0 # convert angles into radians
beta  = 90.0
gamma = 116.0

BRKR_SampleTilt=90.0
BRKR_DetectorTilt=0.0

# Euler Angles (Bunge convention, rotation of the coordinate system base vectors)
BRKR_phi1=17.69
BRKR_Phi=64.56
BRKR_phi2=289.1

# Pattern Center
BRKR_DD =0.701 # in units of screen height
BRKR_PCX=0.508 # in units of screen width, from left of exp. pattern
BRKR_PCY=0.2892 # in units of screen height, from top of exp. pattern

# Screen, only the Aspect Ratio is important here
BRKR_ScreenWidth=320
BRKR_ScreenHeight=230

# generate (hkl) up to (+/-hkl_max +/-hkl_max +/-hkl_max)
hkl_max=8
h_step=2
k_step=4
l_step=4
'''

"""
# albite triclinic example, compare tot orthoclase (monoclinic) example
ExpImageFilename='albite.png'
a =7.107
b =8.153
c =12.869
alpha = 90.3 # convert angles into radians
beta  = 93.5
gamma = 116

BRKR_SampleTilt=90.0
BRKR_DetectorTilt=0.0

# Euler Angles (Bunge convention, rotation of the coordinate system base vectors)
BRKR_phi1=20.63
BRKR_Phi=61.91
BRKR_phi2=286.96

# Pattern Center
BRKR_DD =0.70974 # in units of screen height
BRKR_PCX=0.48366 # in units of screen width, from left of exp. pattern
BRKR_PCY=0.28721 # in units of screen height, from top of exp. pattern

# Screen, only the Aspect Ratio is important here
BRKR_ScreenWidth=320
BRKR_ScreenHeight=230

# generate (hkl) up to (+/-hkl_max +/-hkl_max +/-hkl_max)
hkl_max=8
h_step=2
k_step=4
l_step=4
"""






#------------------------------------------------------------------------------
# NO EXTERNAL PARAMETERS NEEDED BEYOND THIS POINT
#------------------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math as m
import matplotlib.image as image

matplotlib.rcParams.update({'font.size': 16})
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)
#------------------------------------------------------------------------------
# calculation of general parameters from Bruker parameters
#------------------------------------------------------------------------------
alpha = np.deg2rad(alpha) # convert angles into radians
beta  = np.deg2rad(beta)
gamma = np.deg2rad(gamma)

# total x-axis pre-tilt, relative tilt between screen and sample
TotalTilt=((BRKR_SampleTilt-90.0)-BRKR_DetectorTilt) * m.pi/180.0

# Euler Angles in Bunge convention
phi1=float(BRKR_phi1)*m.pi/180.0
Phi =float(BRKR_Phi) *m.pi/180.0
phi2=float(BRKR_phi2)*m.pi/180.0

# limits of gnomonic projection
ScreenAspect=float(BRKR_ScreenWidth)/float(BRKR_ScreenHeight)
y_gn_max= + (    BRKR_PCY)              /BRKR_DD
y_gn_min= - (1.0-BRKR_PCY)              /BRKR_DD
x_gn_max= +((1.0-BRKR_PCX)*ScreenAspect)/BRKR_DD
x_gn_min= -((    BRKR_PCX)*ScreenAspect)/BRKR_DD
#print np.degrees(TotalTilt)
#print(BRKR_PCX,BRKR_PCY,ScreenAspect,BRKR_DD)
#print(y_gn_min,y_gn_max,x_gn_min,x_gn_max)
if PLOT_PATTERN_IMAGE:
    ExperimentalImage = image.imread(ExpImageFilename)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Rotation matrices 
#------------------------------------------------------------------------------
#  see e.g.
#  J. B. Kuipers "Quaternions and Rotation Sequences", 
#  Princeton University Press, 1999

def Rx(RotAngle):
    '''
    provides the X axis (e1) rotation matrix in cartesian systems,
	input "RotAngle" in radians

    meaning for a COLUMN of VECTORS: transformation matrix U for VECTORS
    this matrix rotates a set of "old" basis vectors 
    $(\vec{e1},\vec{e2},\vec{e3})^T$ (column) by +RotAngle (right hand rule) 
    to a new set of basis vectors $(\vec{e1}',\vec{e2}',\vec{e3}')^T$ (column)

    meaning for a COLUMN of vector COORDINATES:
    (1) (N_P_O):    coordinates of fixed vector in a "new" basis that is 
                    rotated by +RotAngle (passive rotation)
    (2) (O_P_O):    active rotation of vector coordinates in the same 
                    "old" basis by -RotAngle
    '''
    mat=np.matrix([ [1,                 0,               0 ],
                    [0,  np.cos(RotAngle), np.sin(RotAngle)],
                    [0, -np.sin(RotAngle), np.cos(RotAngle)]])
    return mat
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def Rz(RotAngle):
    '''
    provides the Z axis (e3) rotation matrix in cartesian systems,
	input "RotAngle" in radians

    meaning for a COLUMN of VECTORS: transformation matrix U for VECTORS
    this matrix rotates a set of "old" basis vectors 
    $(\vec{e1},\vec{e2},\vec{e3})^T$ (column) by +RotAngle (right hand rule) 
    to a new set of basis vectors $(\vec{e1}',\vec{e2}',\vec{e3}')^T$ (column)

    meaning for a COLUMN of vector COORDINATES:
    (1) (N_P_O):    coordinates of fixed vector in a "new" basis that is 
                    rotated by +RotAngle (passive rotation)
    (2) (O_P_O):    active rotation of vector coordinates in the same 
                    "old" basis by -RotAngle
    '''
    mat=np.matrix([[ np.cos(RotAngle) , np.sin(RotAngle), 0 ],
                   [-np.sin(RotAngle) , np.cos(RotAngle), 0 ],
                   [                0 ,                0, 1 ]])
    return mat
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def CalcStructureMatrix(a=1.0,b=1.0,c=1.0,
                        alpha=np.deg2rad(90.0),
                        beta=np.deg2rad(90.0),
                        gamma=np.deg2rad(90.0)):
    '''
    computes the structure matrix from lattice parameters
    input angles in RADIANS    
    convention: McKie&McKie, "Essentials of Crystallography", 1986
    '''

    ca = m.cos(alpha)
    sa = m.sin(alpha)
    cb = m.cos(beta)
    cg = m.cos(gamma)
    
    ax = a * m.sqrt(1.0+2.0*ca*cb*cg-(ca*ca+cb*cb+cg*cg))/sa
    ay = a * (cg-ca*cb)/sa
    az = a * cb
    
    by = b * sa
    bz = b * ca
    
    cz = c
    
    StructureMat=np.matrix([[ax , 0,  0 ],
                            [ay , by, 0 ],
                            [az , bz, cz]])
    return StructureMat
#------------------------------------------------------------------------------


def getPlotRadius(x1,x2,y1,y2):
    '''
    maximum radius of plot from PC from gnomonic pattern extent    
    '''
    PlotR=np.max([x1*x1+y1*y1,x1*x1+y2*y2,x2*x2+y1*y1,x2*x2+y2*y2]) # radius
    #PlotR=np.max(np.abs([x1,x2,y1,y2])) # absolute max. x-y-extent    
    
    PlotR=PATTERN_PLOT_FACTOR*m.sqrt(PlotR)    
    return PlotR

RPlot=getPlotRadius(x_gn_min,x_gn_max,y_gn_min,y_gn_max)




def makeLabelList(bracket1,bracket2,IndexArray):
    '''
    returns the indices formated with chosen bracket
    '''    
    LabelList=list()
    for ix in range(IndexArray.shape[1]):
        formattedline =(bracket1+'%i %i %i'+bracket2) % ( tuple(IndexArray[:,ix]) )
        LabelList.append(formattedline)
        
    return np.array(LabelList)




# reference output
def printRefValues():
    print('phi1,Phi,phi2:')    
    print(phi1,Phi,phi2)
    print(np.degrees(phi1),np.degrees(Phi),np.degrees(phi2))
    print('U_S = Rx(alpha):')
    print(U_S)
    print('Rz(phi1):')
    print(Rz(phi1))
    print('Rx(Phi):')
    print(Rx(Phi))
    print('Rz(phi2):')
    print(Rz(phi2))
    print('U_O = G_sample:')
    print(U_O)
    print('U_O*U_S = G_detector:')
    print(U_O*U_S)
    print('U_K=U_A*U_O*U_S:')
    print(U_K)
    print('Structure Matrix A:')
    print(A)
    vk=np.matrix([1,2,-3]).T #column vector
    print('column vector, direct lattice:')
    print(vk)
    print('column vector, sample frame:')
    print(U_O.T*A*vk)    
    print('column vector, detector frame:')
    print(U_K.T*vk)

#------------------------------------------------------------------------------
# List of generic indices to be used seperately for 
# - direct space (directions) 
# - reciprocal space (plane normals)
#------------------------------------------------------------------------------
# make a simple list of indices pqr
p = np.arange(-hkl_max,hkl_max+1,h_step)
q = np.arange(-hkl_max,hkl_max+1,k_step)
r = np.arange(-hkl_max,hkl_max+1,l_step)
pqr=np.vstack(np.meshgrid(p,q,r)).reshape(3,-1).T
# remove 000 vector
# norms of index vectors
pqr_norms=np.array(np.linalg.norm(pqr, axis=1))
not000=np.less(1e-6,np.abs(pqr_norms))

# lattice plane normals = reciprocal space vectors hkl
# measured in lattice system K ("NEW") = capital letters
HKL=pqr[not000].T

# direct space vectors = zone axes
UVW=pqr[not000].T

# save indices to files
#np.savetxt('HKL.txt',HKL.T,fmt='%4i')
#np.savetxt('UVW.txt',UVW.T,fmt='%4i')
#------------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# SET UP TRANSFORMATION MATRICES
# -----------------------------------------------------------------------------

# STRUCTURE MATRICES
#------------------------------------------------------------------------------
# direct structure matrix A
# A gives the direct lattice basis vector coordinates (column vectors)
# in the Cartesian system
A = CalcStructureMatrix(a,b,c,alpha,beta,gamma)
# reciprocal structure matrix A+: transpose of inverse of A
# A+ gives to reciprocal lattice basis vector coordinates (column vectors)
# in the Cartesian system
Aplus = (A.I).T
#------------------------------------------------------------------------------

# rotation from Detector system to sample system
U_S = Rx(TotalTilt)


# rotation of Sample System to Cartesian Crystal System = ORIENTATION OF GRAIN
U_O = Rz(phi2)  * Rx(Phi) * Rz(phi1)


# transformation of Cartesian Crystal system 
U_A     = A.T       # to direct Crystal lattice K
U_Astar = Aplus.T   # to reciprocal lattice Kstar

U_K_M     =U_A     * U_O * U_S
U_Kstar_M =U_Astar * U_O * U_S

# ---- TOTAL TRANSFORMATION MATRICES-------------------------------------------
# as numpy arrays (no np.matrix anymore from here on)
U_K     = np.asarray(U_K_M )
U_Kstar = np.asarray(U_Kstar_M)
U_KstarIT=np.asarray(U_Kstar_M.I.T)
# -----------------------------------------------------------------------------

#printRefValues()

# -----------------------------------------------------------------------------
# TRANSFORMATION OF UVW direct space vectors to detector system
# RULE: "old" (detector) from "NEW" (direct lattice) coordinates by U^T
# -----------------------------------------------------------------------------
UVW_D = np.dot(U_K.T, UVW) # column vectors UVW
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# TRANSFORMATION OF HKL reciprocal space vectors to detector system 
# RULE: "old" (detector) from "NEW" (recipr. lattice) by U^T (here: U_KStar^T)
# -----------------------------------------------------------------------------
HKL_D = np.dot(U_Kstar.T, HKL) # column vectors UVW.T
# -----------------------------------------------------------------------------

#np.savetxt('HKL_D.txt',HKL_D.T,fmt='%13.6f')
#np.savetxt('UVW_D.txt',UVW_D.T,fmt='%13.6f')

# -----------------------------------------------------------------------------
# Calculate geometric data for plotting the gnomonic projection
# -----------------------------------------------------------------------------

# UVW directions
# project the detector coordinates to the gnomonic plane
UVW_x,UVW_y,UVW_z = UVW_D.T[:,0], UVW_D.T[:,1], UVW_D.T[:,2]
#UVW_r      = np.sqrt(UVW_x**2+UVW_y**2+UVW_z**2)
#UVW_phi    = np.arctan2(UVW_y,UVW_x)
#UVW_theta  = np.arccos(UVW_z/UVW_r)
UVW_isUpperHemi= np.greater(UVW_z,0.0)

# only take upper hemisphere
UVW_x_u = UVW_x[UVW_isUpperHemi]
UVW_y_u = UVW_y[UVW_isUpperHemi]
UVW_z_u = UVW_z[UVW_isUpperHemi]
UVW_u   = UVW.T[UVW_isUpperHemi]

#print 'UVW_u'
#print UVW_u

# Gnomonic coordinates in detector system
UVW_x_gn=UVW_x_u/UVW_z_u
UVW_y_gn=UVW_y_u/UVW_z_u
UVW_Labels=makeLabelList('[',']',UVW_u.T)

# HKL plane poles
# project the detector coordinates to the gnomonic plane
HKL_x,HKL_y,HKL_z = HKL_D.T[:,0], HKL_D.T[:,1], HKL_D.T[:,2]
HKL_r      = np.sqrt(HKL_x**2+HKL_y**2+HKL_z**2)
HKL_phi    = np.arctan2(HKL_y, HKL_x)
HKL_theta  = np.arccos(HKL_z/HKL_r)

HKL_isUpperHemi= np.greater(HKL_z,0.0)
# only take upper hemisphere
HKL_x_u = HKL_x[HKL_isUpperHemi]
HKL_y_u = HKL_y[HKL_isUpperHemi]
HKL_z_u = HKL_z[HKL_isUpperHemi]
HKL_u   = HKL.T[HKL_isUpperHemi]

# Gnomonic coordinates in detector system
HKL_x_gn=HKL_x_u/HKL_z_u
HKL_y_gn=HKL_y_u/HKL_z_u
HKL_Pole_Labels=makeLabelList('(',')',HKL_u.T)

#np.savetxt('Gnom.txt', [HKL_x_gn.T,HKL_y_gn.T]  )


#------------------------------------------------------------------------------
# plotting of lattice plane traces by finding Hessian normal form of lines
#------------------------------------------------------------------------------
# calculating the lattice plane traces in gnomonic projection
R_Hesse=5.0*RPlot # =tan(theta_max) circle for getting lattice plane traces as coords
# find Hesse normal form of lattice plane trace
# distance from origin is right-angle (Pi/2) complement to distance of pole
d_Hesse=np.tan(0.5*np.pi-HKL_theta)
inCircle=np.less(np.abs(d_Hesse),R_Hesse)
isFullUpper=np.greater(HKL_z,-0.00001) # to include z=0 for traces 

inCircle=np.logical_and(inCircle,isFullUpper)

HKL_InCircle=HKL.T[inCircle]
#HKL_InCircle=HKL.T[isPlotTrace]

HKL_Trace_Labels=makeLabelList('(',')',HKL_InCircle.T)

# angle from PC (0,0) to point on circle where the line cuts the circle
alpha_Hesse=np.arccos(d_Hesse[inCircle]/R_Hesse)

alpha1_hkl=HKL_phi[inCircle]-np.pi+alpha_Hesse
alpha2_hkl=HKL_phi[inCircle]-np.pi-alpha_Hesse

C1x=R_Hesse*np.cos(alpha1_hkl)
C1y=R_Hesse*np.sin(alpha1_hkl)

C2x=R_Hesse*np.cos(alpha2_hkl)
C2y=R_Hesse*np.sin(alpha2_hkl)

# label coordinates on smaller circle within plot
R_HESSE_LABELS=0.925*RPlot
R_Hesse=R_HESSE_LABELS

alpha_Hesse=np.arccos(d_Hesse[inCircle]/R_Hesse)

alpha1_hkl=HKL_phi[inCircle]-np.pi+alpha_Hesse
alpha2_hkl=HKL_phi[inCircle]-np.pi-alpha_Hesse

C1xLabels=R_Hesse*np.cos(alpha1_hkl)
C1yLabels=R_Hesse*np.sin(alpha1_hkl)

C2xLabels=R_Hesse*np.cos(alpha2_hkl)
C2yLabels=R_Hesse*np.sin(alpha2_hkl)


# POLE LINES: projections of (hkl) plane normal connected to trace
# only for upper hemisphere
HPointsX=-d_Hesse*np.cos(HKL_phi)
HPointsY=-d_Hesse*np.sin(HKL_phi)

HPointsX_u=HPointsX[HKL_isUpperHemi]
HPointsY_u=HPointsY[HKL_isUpperHemi]
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# do the PLOTTING using matplotlib
# -----------------------------------------------------------------------------

# size of symbols
PoleSizePt = 6
area = np.pi * PoleSizePt**2


#  set plot size
#fig, ax= plt.subplots()
fig=plt.figure
#plt.figure(figsize=(10.0*ScreenAspect, 10.0)) # rectangualr canvas
plt.figure(figsize=(10.0, 10.0))
#plt.xlim(x_gn_min,x_gn_max)
#plt.ylim(y_gn_min,y_gn_max)
plt.set_cmap('gray')
plt.xlabel('x / z',fontsize=28)
plt.ylabel('y / z',fontsize=28)
#v = 1.5*R_Hesse*np.array([-1, 1, -1, 1]) #v = [xmin, xmax, ymin, ymax]
v = RPlot*np.array([-1, 1, -1, 1]) #v = [xmin, xmax, ymin, ymax]
plt.axis(v)
# plt.axes().set_aspect('equal', 'datalim') # datalim to fill whole canvas
plt.axes().set_aspect('equal') # square plot area

# show experimental pattern in background
if PLOT_PATTERN_IMAGE:
    plt.imshow(ExperimentalImage,  #aspect=1.0,
           extent=(x_gn_min,x_gn_max,y_gn_min,y_gn_max), alpha=0.75, zorder=-1)




#------------------------------------------------------------------------------
# draw circles for angular distances from Pattern Center
if (PLOT_GNOM_THETA==True):
    for ThetaGnom in range(0,81,10):       
        circle1=plt.Circle((0,0), m.tan(np.deg2rad(ThetaGnom)),
                       alpha=0.15,edgecolor='k',facecolor='None',linewidth=3)
        plt.gcf().gca().add_artist(circle1)     
#------------------------------------------------------------------------------        


        
# plot LATTICE PLANE TRACES
plt.plot([C1x, C2x], [C1y, C2y], color='blue', linestyle='-', linewidth=1.0,alpha=0.4,zorder=80)


# ZONE AXES
plt.scatter(UVW_x_gn, UVW_y_gn, s=area, c='yellow', alpha=1.0, zorder=90)


if (PLOT_HKL_POLES==True):
    # lattice plane POLES with PENTAGONS (indicates that this cannot be observed directly)
    plt.scatter(HKL_x_gn, HKL_y_gn, marker='p', s=2.5*area, c='red', alpha=0.8,zorder=80)

    # POLE labels
    labels = HKL_Pole_Labels
    for label, x, y in zip(labels, HKL_x_gn, HKL_y_gn):
        plt.annotate(
        label, zorder=100,
            xy = (x, y), xytext = (0, -14), fontsize=8,
            textcoords = 'offset points', ha = 'center', va = 'top',
            color='black', weight='bold',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'red', alpha = 0.4),
            arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    if (PLOT_HESSE_LINES==True):
        # plot projections of plane normals from Hesse point to Plane Pole
        plt.plot([HKL_x_gn, HPointsX_u], [HKL_y_gn, HPointsY_u], color='red', linestyle='--', linewidth=1.5,alpha=0.8,zorder=60)
        # "Hesse points": meet corresponding (hkl) trace at right angle
        plt.scatter(HPointsX_u, HPointsY_u, s=0.3*area, c='red', marker='s', alpha=1.0, zorder=10)


# ------------ LABELS ----------------------------------------------
# PLANE TRACE labels 
labels = HKL_Trace_Labels
for label, x, y in zip(labels, C1xLabels, C1yLabels):
    plt.annotate(
        label,zorder=110,
        xy = (x, y), xytext = (0,0), fontsize=6,
        textcoords = 'offset points',
        ha = 'center', va = 'bottom', color='white', weight='bold',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'blue',alpha = 0.7),
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))


# ZONE AXIS labels
labels = UVW_Labels[:] #[UVW_isUpperHemi]
for label, x, y in zip(labels, UVW_x_gn, UVW_y_gn):
    plt.annotate(
        label, zorder=105,
        xy = (x, y), xytext = (0, 14), fontsize=6,
        color='black', weight='bold',
        textcoords = 'offset points', ha = 'center', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.7),
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))

# Pattern Center
plt.scatter(0, 0, marker='*',s=1.5*area, alpha=0.7, facecolor='red', edgecolor='k', zorder=200)
plt.annotate(
        'PC',
        xy = (0, 0), xytext = (0,14), fontsize=10,
        textcoords = 'offset points', zorder=201,
        ha = 'center', va = 'bottom',
        color='white', weight='black',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'red', alpha = 1.0),
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
plt.savefig(ExpImageFilename+"_output.png", dpi=600) # uncomment to save the result
plt.show()
#------------------------------------------------------------------------------
'''
phi1,Phi,phi2:
(2.2113321622768156, 0.6614797865058508, 4.750786223928564)
(126.7, 37.899999999999999, 272.19999999999999)
U_S = Rx(alpha):
[[ 1.          0.          0.        ]
 [ 0.          0.90938136 -0.41596338]
 [ 0.          0.41596338  0.90938136]]
Rz(phi1):
[[-0.59762515  0.80177564  0.        ]
 [-0.80177564 -0.59762515  0.        ]
 [ 0.          0.          1.        ]]
Rx(Phi):
[[ 1.          0.          0.        ]
 [ 0.          0.78908408  0.6142852 ]
 [ 0.         -0.6142852   0.78908408]]
Rz(phi2):
[[ 0.03838781 -0.99926292  0.        ]
 [ 0.99926292  0.03838781  0.        ]
 [ 0.          0.          1.        ]]
U_O = G_sample:
[[ 0.60926055  0.50200731 -0.61383242]
 [-0.6214714   0.78308188  0.02358106]
 [ 0.49251891  0.36711228  0.78908408]]
U_O*U_S = G_detector:
[[ 0.60926055  0.20118428 -0.76702442]
 [-0.6214714   0.72192893 -0.30428921]
 [ 0.49251891  0.66207515  0.56487309]]
U_K=U_A*U_O*U_S:
[[ 0.60926055  0.20118428 -0.76702442]
 [-0.6214714   0.72192893 -0.30428921]
 [ 0.49251891  0.66207515  0.56487309]]
Structure Matrix A:
[[  1.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  6.12323400e-17   1.00000000e+00   0.00000000e+00]
 [  6.12323400e-17   6.12323400e-17   1.00000000e+00]]
column vector, direct lattice:
[[ 1]
 [ 2]
 [-3]]
column vector, sample frame:
[[-2.11123899]
 [ 0.96683422]
 [-2.93392255]]
column vector, detector frame:
[[-2.11123899]
 [-0.34118333]
 [-3.07022212]]
 '''