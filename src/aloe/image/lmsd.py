import numpy as np
import numba
import math
from numba import njit, prange, set_num_threads, float64

@njit(float64(float64, float64))
def max_float(a: float, b: float) -> float:
    return a if a > b else b

def local_normalization_numba_mirror(img, r):
    """
    Applies local normalization to an input image by subtracting the local mean and dividing by the local standard deviation
    within a square window of radius r around each pixel, explicitly mirroring pixels beyond the edges using Numba.

    Parameters:
    img (numpy.ndarray): Input 2D array of shape (nrows, ncols).
    r (int): Radius of the neighborhood around each pixel.

    Returns:
    numpy.ndarray: The locally normalized image.
    """
    nrows, ncols = img.shape
    normalized_img = np.zeros_like(img)
    
    set_num_threads(48)
    @njit(parallel=True)
    def compute_normalization(img, normalized_img, r, nrows, ncols):
        for i in prange(nrows):
            for j in range(ncols):
                sum_val = 0.0
                sum_sq_val = 0.0
                count = 0
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        ni = i + dy
                        nj = j + dx
                        # Mirror indices beyond edges
                        if ni < 0:
                            ni = -ni - 1
                        elif ni >= nrows:
                            ni = 2*nrows - ni - 1
                        if nj < 0:
                            nj = -nj - 1
                        elif nj >= ncols:
                            nj = 2*ncols - nj - 1
                        val = img[ni, nj]
                        sum_val += val
                        sum_sq_val += val * val
                        count += 1
                mean = sum_val / count
                variance = (sum_sq_val / count) - mean * mean
                variance = max_float(variance, 0.0)
                std = math.sqrt(variance) + 1e-8  # Add epsilon to prevent division by zero
                normalized_img[i, j] = (img[i, j] - mean) / std

    compute_normalization(img, normalized_img, r, nrows, ncols)

    return normalized_img



def local_normalization_sliding_window(img, r):
    """
    Applies local normalization to an input image by subtracting the local mean and dividing by the local standard deviation
    within a square window of radius r around each pixel, reusing computations from neighboring pixels using Numba.
    
    Parameters:
    img (numpy.ndarray): Input 2D array of shape (nrows, ncols).
    r (int): Radius of the neighborhood around each pixel.
    
    Returns:
    numpy.ndarray: The locally normalized image.
    """
    nrows, ncols = img.shape
    normalized_img = np.zeros_like(img)

    # Pad the image with mirrored edges
    padded_img = np.pad(img, r, mode='reflect')
    
    set_num_threads(24)
    @njit
    #@njit(parallel=True)
    def compute_normalization(padded_img, normalized_img, r, nrows, ncols):
        window_size = 2 * r + 1
        count = window_size * window_size
        for i in range(nrows):
            # Initialize sums for the first window in the row
            sum_val = 0.0
            sum_sq_val = 0.0
            for dy in range(window_size):
                for dx in range(window_size):
                    val = padded_img[i + dy, dx]
                    sum_val += val
                    sum_sq_val += val * val
            # Compute mean and std for the first window in the row
            mean = sum_val / count
            variance = (sum_sq_val / count) - mean * mean
            variance = max_float(variance, 0.0)
            std = np.sqrt(variance) + 1e-8
            normalized_img[i, 0] = (img[i, 0] - mean) / std

            # Slide the window across the row
            for j in range(1, ncols):
                # Subtract the leftmost column
                for dy in range(window_size):
                    val = padded_img[i + dy, j - 1]
                    sum_val -= val
                    sum_sq_val -= val * val
                # Add the new rightmost column
                for dy in range(window_size):
                    val = padded_img[i + dy, j + window_size - 1]
                    sum_val += val
                    sum_sq_val += val * val
                # Compute mean and std
                mean = sum_val / count
                variance = (sum_sq_val / count) - mean * mean
                variance = max_float(variance, 0.0)
                std = np.sqrt(variance) + 1e-8
                normalized_img[i, j] = (img[i, j] - mean) / std

    compute_normalization(padded_img, normalized_img, r, nrows, ncols)

    return normalized_img
