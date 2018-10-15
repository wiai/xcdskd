import numpy as np

def reimer2bf(theta,thetaBragg,K,sgscale,p1,p2):
    """ full two-term two beam Kikuchi band profile as function of 
        theta=angle from lattice plane (middle of Kikuchi band)
        
        K : 1/wavelength
        gx: reciprocal space vector perpendicular to lattice plane
        zetag : extinction distance
        zeta0s,zetags: absorption parameters
            
        see: Reimer, SEM p. 351 eq. 9.45
        
    """
    kx  =  K*np.sin(theta)
    gx  =  2.0*K*np.sin(thetaBragg)
    sg  =  (2.0*kx*gx - gx**2)/(2.0*K)
    w   =  sgscale*sg
    prof1_int  = - (w + p1) / (1.0+w**2-p1**2)
    prof2_int  = + w / (1.0+w**2+((1.0+w**2)*p2)**2)
 
    return w, prof1_int+prof2_int
    
def kikuprofile(theta,thetaBragg,K,sharpness=20.0,edgedarkness=0.6,rounding=100.0,asymmetry=1.0):
    """ model for Kikuchi band profile
    """
    w2,profile_right = reimer2bf(theta, thetaBragg,K,sharpness,edgedarkness,rounding)
    w1,profile_left  = reimer2bf(theta,-thetaBragg,K,sharpness,edgedarkness,rounding)
    return w1,profile_right+asymmetry*profile_left

if __name__=="__main__":
    import matplotlib
    matplotlib.use("Qt5Agg")
    #matplotlib.use("Qt5Agg", force=true) # force PyQt5
    import matplotlib.pyplot as plt
    print("Reimer two-beam example profile")
    theta=np.linspace(-15,15,300)*np.pi/180
    #gx=2.0*np.pi/3.0 # 3 Angstroem lattice constant
    K=1.0/0.085885
    thetaBragg=5.0*np.pi/180.0
    w,profile=kikuprofile(theta,thetaBragg,K,sharpness=30.0,
        edgedarkness=0.95,rounding=1,asymmetry=1.1) 
    #plt.plot(theta*180.0/np.pi,profile) 
    plt.plot(np.sin(theta)/np.sin(thetaBragg),profile)
    #plt.plot(w2,profile)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    plt.show()
    #print(profile)