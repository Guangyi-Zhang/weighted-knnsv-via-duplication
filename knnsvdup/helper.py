import numpy as np


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