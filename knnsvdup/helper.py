import numpy as np


def approx_harmonic_sum(j):
    """
    Approximate the sum of harmonic series from 1 to j using the Euler-Maclaurin formula.
    
    H_j ≈ ln(j) + gamma + 1/(2j) - 1/(8j^2)
    
    where gamma ≈ 0.5772 is the Euler-Mascheroni constant.
    
    Args:
        j: Upper limit of the harmonic series
        
    Returns:
        Approximation of the harmonic sum
    """
    if j < 1000: # return the real sum if j is small
        return sum(1/j for j in range(1, j+1))
    
    gamma = 0.5772156649  # Euler-Mascheroni constant

    real1000 = 7.485470860550343 # the real sum of the first 1000 terms
    apx1000 = np.log(1000) + gamma + 1/(2*1000) - 1/(8*1000**2)
    
    apx = np.log(j) + gamma + 1/(2*j) - 1/(8*j**2)

    # the approx is not accurate for small j, so we adjust it by replacing the first 1000 terms with the real sum
    return apx + (real1000 - apx1000) 


def distance(x, y):
    """
    Compute the Euclidean distance between two points.
    """
    return np.linalg.norm(x - y)


def kernel_value(d, sigma):
    """
    Compute the Gaussian kernel value for a given distance.
    """
    return np.exp(-d**2 / (2 * sigma**2))    