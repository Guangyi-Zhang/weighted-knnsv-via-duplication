import numpy as np

from knnsvdup.helper import distance, kernel_value


def shapley_unweighted_bf(D, Z_test, K):
    """
    Compute Shapley values for unweighted KNN using brute force.
    """
    if not isinstance(Z_test, list):
        Z_test = [Z_test]
    
    n_test = len(Z_test)
    shapley_values = np.zeros(len(D))
    for i in range(n_test):
        s = shapley_unweighted_bf_single(D, Z_test[i], K)
        shapley_values += s

    return shapley_values / n_test
    

def shapley_unweighted_bf_single(D, z_test, K):
    """
    Compute Shapley values for unweighted KNN using recursive formula.
    
    Args:
        D: List of tuples (x, y) where x is feature vector, y is label
        z_test: Test point tuple (x_test, y_test)
        K: Number of neighbors for KNN
        
    Returns:
        Array of Shapley values for each data point
    """
    x_test, y_test = z_test
    n = len(D)
    if n == 0:
        return np.array([])
    
    # Calculate distances and sort
    dxy = [(distance(x, x_test), x, y) for x, y in D]
    dxy_idx = list(range(len(dxy)))    
    sorted_dxy_idx = sorted(dxy_idx, key=lambda i: dxy[i][0]) # argsort
    
    # Extract label matches
    y_match = [1 if dxy[i][2] == y_test else 0 for i in sorted_dxy_idx]
    
    # Initialize Shapley values array
    s = np.zeros(n)
    
    # Base case: farthest point
    idx_n = sorted_dxy_idx[-1]
    s[idx_n] = (K/len(sorted_dxy_idx)) * y_match[-1]
    
    # Recursive calculation from 2nd farthest to nearest
    for j in range(len(sorted_dxy_idx)-2, -1, -1):
        i_plus_1 = j + 1  # Convert to 1-based index
        idx_j = sorted_dxy_idx[j]
        idx_j_plus_1 = sorted_dxy_idx[j+1]
        term = (min(K, i_plus_1)/i_plus_1) * (
            y_match[j] - y_match[j+1]
        )
        s[idx_j] = s[idx_j_plus_1] + term
        
    return s 