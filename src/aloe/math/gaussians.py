"""
functions for Gaussian peaks
"""

import numpy as np

def gaussian_fwhm(x, x0, fwhm):
    """ Gaussian window peak defined by FWHM """
    return np.exp(-4.0*np.log(2.0)*np.power( (x - x0)/fwhm, 2.0))
  
