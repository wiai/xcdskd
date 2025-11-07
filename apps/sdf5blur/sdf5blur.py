"""
Treatment of the observed angular resolution of experimental Kikuchi diffraction patterns
from non-perfect materials

Based on Gaussian smoothing of the SDF5 master hemispheres with sigma = sigma_deg
"""

# original unblurred file
filename_sdf5 = "Mo-Molybdenum_15kV_2001.sdf5"

# standard deviation width parameter for angular Gaussian blur
sigma_deg = 0.2


import numpy as np
import math as m
from tqdm import tqdm
import h5py
import numba
from numba import njit, prange
import shutil
import os


def fstr(number):
    """ return encoded float number for filename 
    with f for decimal point
    """
    return str(number).replace('.', 'f')


@numba.jit(nopython=True)
def orthogonal_unitvector(v):
    # https://stackoverflow.com/questions/11132681/what-is-a-formula-to-get-a-vector-perpendicular-to-another-vector
    # if |c|<|a| then return (b,-a,0) else return (0,-c,b)
    if (abs(v[2]) < abs(v[0])):
        ortho = np.array((v[1], -v[0], 0.0))
    else:
        ortho = np.array((0.0 , -v[2], v[1]))
    ortho /= np.linalg.norm(ortho)
    return ortho


@numba.jit(nopython=True)
def calc_weights(sigmaRad, nsteps):

    stepSize = 3.0 * sigmaRad / nsteps; # filter extension up to 3*sigma
    expScale = (stepSize **2) / (2.0 * sigmaRad**2); # decay constant

    #precalculate the filter weights
    arraySize = (2 * nsteps) + 1;
    weights = np.zeros((arraySize, arraySize), dtype=np.float64);
    sumWeight = 0.0;
    for ix in range(-nsteps, nsteps + 1, 1):
        for iy in range(-nsteps, nsteps + 1, 1):
            weight = np.exp(-expScale * ((ix * ix) + (iy * iy)));
            weights[ix + nsteps, iy + nsteps] = weight;
            sumWeight += weight;

    # normalize weights
    weights /= sumWeight;
    return weights


@numba.jit(nopython=True)
def normalize(v):
    return v/np.linalg.norm(v)


@numba.jit(nopython=True)
def get_intensity(hemi_upper, hemi_lower, v):
    """ get nearest pixel from hemispheres
    """
    x,y,z = v
    icolmax = hemi_upper.shape[0]-1
    hemi = hemi_upper
    if (z<0):
        z *= -1
        hemi = hemi_lower
    
    xg = x / (1+z)
    yg = y / (1+z)
    
    # x along cols, from -1 up to +1
    x0 = 0.5 * (xg + 1.0) * icolmax
    # y along rows, from +1 down to -1
    y0 = 0.5 * (1.0 - yg) * icolmax
    
    c0 = int(x0)
    r0 = int(y0)
    
    if (c0<0): c0=0
    if (r0<0): r0=0
    
    if (c0>icolmax): c0=icolmax
    if (r0>icolmax): r0=icolmax
    
    c1 = c0+1
    if (c1>icolmax): c1=icolmax
    r1 = r0+1
    if (r1>icolmax): r1=icolmax
    
    A = float(hemi[r0, c0])
    B = float(hemi[r0, c1])
    C = float(hemi[r1, c0])
    D = float(hemi[r1, c1])

    ax = float(x0 - c0)
    by = float(y0 - r0)

    E = A + ax * (B-A) # row 0, along col + 1
    F = C + ax * (D-C) # row 1, along col + 1

    G = E + by * (F-E) 
    
    return G


@numba.jit(nopython=True)
def get_averaged_intensity(vCenter, nsteps, sigmaRad, weights, hemi_upper, hemi_lower):

    stepSizeRad = 3.0 * sigmaRad / nsteps; # filter extension up to 3*sigma 

    # make vector grid near vCenter
    vZ = normalize(vCenter)

    # 2 vectors in a plane perpendicular to vCenter, steps in 2D will be in sin(angle) approx angle_rad
    vX = orthogonal_unitvector(vZ)
    vY = normalize(np.cross(vZ, vX))

    result = 0.0
    for ix in range(-nsteps, nsteps + 1, 1):
        for iy in range(-nsteps, nsteps + 1, 1):
            vSample = normalize(vZ + vX * ix * stepSizeRad + vY * iy * stepSizeRad)
            result += weights[ix + nsteps, iy + nsteps] * get_intensity(hemi_upper, hemi_lower, vSample)

    return result;


@njit(parallel=True)
def blur_hemi(hemi_upper, hemi_lower, ihemi_blur, sigma_deg):
    """ Gaussian blur (in angle space) of stereographic hemisphere
    """
    nsteps = 15 # +/- steps around reference direction
    sigma_rad = np.radians(sigma_deg)
    weights = calc_weights(sigma_rad, nsteps)

    
    hemi_blurred = np.zeros_like(hemi_upper);
    nrows, ncols = hemi_upper.shape 
    icenter = (ncols - 1) / 2;

    # sample hemisphere vectors
    for irow in prange(nrows):      
        for icol in range(ncols):
            xs = +1.0 * (icol - icenter) / icenter; # xs in stereographic master array is increasing with column number
            ys = -1.0 * (irow - icenter) / icenter; # ys in stereographic master array is decreasing with row number

            # get 3D vector (x,y,z) from 2D projection (xs, ys)
            # https://math.stackexchange.com/questions/2652532/understanding-the-formula-for-stereographic-projection-of-a-point
            length = 1.0 + (xs * xs) + (ys * ys);
            x = 2.0 * xs / length;
            y = 2.0 * ys / length;
            z = (1.0 - (xs * xs) - (ys * ys)) / length; # for the upper hemisphere we need the negative z of formula in link

            if (ihemi_blur == 1):
                # z is from lower hemisphere
                z *= -1.0;

            vstereo = np.array((x, y, z))
            hemi_blurred[irow, icol] = get_averaged_intensity(vstereo, nsteps, sigma_rad, weights, hemi_upper, hemi_lower);

    return hemi_blurred


def blur_hemi_datasets(h5, simtype="Dynamical", sigma_deg=0.25):
    master_group = "/Data/Master/"
    dataset_upper = master_group + '/' + simtype + '/Upper'
    hemi_upper = np.copy(h5[dataset_upper])
    
    dataset_lower = master_group + '/' + simtype + '/Lower'
    hemi_lower = np.copy(h5[dataset_lower])
    
    hemi_blurred_upper = blur_hemi(hemi_upper, hemi_lower, 0, sigma_deg)
    hemi_blurred_lower = blur_hemi(hemi_upper, hemi_lower, 1, sigma_deg)

    h5[dataset_upper][...] = hemi_blurred_upper[...]
    h5[dataset_lower][...] = hemi_blurred_lower[...]
    
    return 


def main():
    print("Input file:", filename_sdf5)

    filename_sdf5_blurred = os.path.splitext(os.path.basename(filename_sdf5))[0]+ "_blurred_"+fstr(sigma_deg)+"deg.sdf5"
    shutil.copy(filename_sdf5, filename_sdf5_blurred)
    print("Output file:", filename_sdf5_blurred)
    print("Processing file, please wait...")
    h5 = h5py.File(filename_sdf5_blurred, "r+") # r+ read/write, file must exist
    blur_hemi_datasets(h5, simtype="Dynamical", sigma_deg=sigma_deg)
    blur_hemi_datasets(h5, simtype="Kinematic", sigma_deg=sigma_deg)
    blur_hemi_datasets(h5, simtype="TwoBeam", sigma_deg=sigma_deg)
    h5.close()

    return
    
if __name__ == "__main__":
    main()