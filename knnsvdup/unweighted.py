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
    if n == 0 or n == 1 or n < K: # assume n >= 2 and n >= K
        return np.array([])
    
    # Calculate distances and sort
    dxy = [(distance(x, x_test), x, y) for x, y in D]
    dxy_idx = list(range(len(dxy)))    
    sorted_dxy_idx = sorted(dxy_idx, key=lambda i: dxy[i][0]) # argsort
    
    # Extract label matches (1 if label matches test point, 0 otherwise)
    y_match = [1 if dxy[i][2] == y_test else 0 for i in sorted_dxy_idx]
    
    # Initialize Shapley values array
    s = np.zeros(n)
    
    # Compute number of classes (assuming C classes)
    unique_labels = set(y for _, y in D)
    C = len(unique_labels)
    
    # Base case: compute for the farthest point (n-th point)
    idx_n = sorted_dxy_idx[-1]
    
    # Compute sum_{j=1}^{n-1} 1/(j+1)
    harmonic_sum1 = sum(1/(j+1) for j in range(1, n))
    harmonic_sum2 = 1 + harmonic_sum1 # equals to sum(1/j for j in range(1, n+1))
    
    # Average label match for the first n-1 points
    avg_match_n_minus_1 = sum(y_match[:-1]) / (n-1)
    
    # Compute base case s_n
    s[idx_n] = (1/n) * (y_match[-1] - avg_match_n_minus_1) * harmonic_sum1 + (y_match[-1] - 1/C) / n
    
    # Recursive calculation from 2nd farthest to nearest
    for j in range(n-2, -1, -1):
        i = j + 1  # Convert to 1-based index
        idx_i = sorted_dxy_idx[j]
        idx_i_plus_1 = sorted_dxy_idx[j+1]
        
        # Compute the adjustment term
        adjustment = (1/K) * ((min(K, i) * (n-1) / i) - K)
        
        # Compute the difference in recursive formula
        term = (y_match[j] - y_match[j+1]) / (n-1) * (harmonic_sum2 + adjustment)
        
        s[idx_i] = s[idx_i_plus_1] + term
        
    return s 