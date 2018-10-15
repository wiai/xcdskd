import math
import numpy as np

def wavelength_e(Ekin):
    """ calculates relativistically correct electron wavelength 
    
    input: energy in keV
    output: wavelength in Angstroms
    
    Notes:
    
    based on formula for relativistic kinetic energy
    Ekin=sqrt((m*c**2)**2+(p*c)**2)-m*c**2
    and
    lambda=h/p
    """
    hbarc=1.97327052356 # hbar*c in keV*AA
    mec2=510.999064505 # rest mass of electron in keV
    
    pc=math.sqrt((Ekin+mec2)**2-(mec2*mec2))
    lam=(2.0*math.pi*hbarc)/pc
    
    return lam
    
def XWavel(energy_keV):
    """ calculates photon wavelength 
 
    input: energy in keV
    output: wavelength in Angstroms
    
    Notes:
    E=hc/lambda
    h*c is 1239.84 eV*nm
    """
    hc = 1239.841842144513
    lam = 10.0 * hc / (1000.0*energy_keV)
    return lam    


    
def bragg_angle(dhkl,wavel):
    """ Bragg law: lamda = 2*d*sin(theta) 
    """
    return np.arcsin(wavel/(2.0*dhkl))


    
if __name__=="__main__":
    # 0.1keV electron= 1.2263662786039002 \AA
    print("Electron wavelengths:")
    print("100eV: ", eWavel(0.1) , " Angstr.")
    print("10keV: ", eWavel(10.0), " Angstr.")
    print("15keV: ", eWavel(15.0), " Angstr.")
    print("18keV: ", eWavel(18.0), " Angstr.")
    print("20keV: ", eWavel(20.0), " Angstr.")
    print("38keV: ", eWavel(38), " Angstr.")
    
    print("Photon wavelengths:")
    print("100eV: ", XWavel(0.1) , " Angstr.")
    print("20keV: ", XWavel(20.0), " Angstr.")
    print("200keV: ", XWavel(200.0), " Angstr.")
    print("100keV: ", XWavel(100.0), " Angstr.")
    
