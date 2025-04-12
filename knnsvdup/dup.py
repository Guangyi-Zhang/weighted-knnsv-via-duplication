import numpy as np
import math

from knnsvdup.helper import distance, approx_harmonic_sum


def shapley(D, Z_test, K, kernel_fn=None, scaler=1e8, n_perms=None):
    """
    Compute KNN Shapley values for multiple test points.
    """
    if not isinstance(Z_test, list):
        Z_test = [Z_test]
    
    n_test = len(Z_test)
    shapley_values = np.zeros(len(D))
    for i in range(n_test):
        if kernel_fn is None:
            s = shapley_unweighted_single(D, Z_test[i], K)
        elif n_perms is None:
            s = shapley_dup_single(D, Z_test[i], K, kernel_fn, scaler)
        else:
            s = shapley_mc_single(D, Z_test[i], K, kernel_fn, n_perms)
        shapley_values += s

    return shapley_values / n_test
    

def shapley_unweighted_single(D, z_test, K):
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


def shapley_dup_single(D, z_test, K, kernel_fn, scaler=1e8):
    """
    Compute Shapley values for weighted KNN using duplication.
    
    Args:
        D: List of tuples (x, y) where x is feature vector, y is label
        z_test: Test point tuple (x_test, y_test)
        K: Number of neighbors for KNN
        kernel_fn: Kernel function to compute weights

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
    
    # Extract weights and scale to integers
    w_real = [kernel_fn(dxy[i][0]) for i in sorted_dxy_idx]
    w = [math.ceil(w_ * scaler) for w_ in w_real] # rounding to 1 if w < 1/scaler, so n' > K'
    
    # Calculate n' = sum of all weights
    n_prime = sum(w)
    
    # Calculate K' = sum of weights for K nearest neighbors
    K_prime = sum(w[:K])

    # Extract label matches (1 if label matches test point, 0 otherwise)
    y_match = [1 if dxy[i][2] == y_test else 0 for i in sorted_dxy_idx]
    
    # Initialize Shapley values array
    s = np.zeros(n)
    
    # Compute number of classes (assuming C classes)
    unique_labels = set(y for _, y in D)
    C = len(unique_labels)
    
    # Base case: compute for the farthest point (n-th point)
    idx_n = sorted_dxy_idx[-1]
    weight_n = w[-1]
    
    # Compute sum_{j=1}^{n'-1} 1/(j+1)
    harmonic_sum = approx_harmonic_sum(K_prime)
    harmonic_sum_minus_1 = harmonic_sum - 1
    
    # Average weighted label match for the first n-1 points
    weighted_sum_others = sum(w[i] * y_match[i] for i in range(n-1))
    avg_match_term = ((weight_n - 1) * y_match[-1] + weighted_sum_others) / (n_prime - 1)
    
    # Compute base case s_n
    s[idx_n] = ((y_match[-1] - avg_match_term) * harmonic_sum_minus_1 + (y_match[-1] - 1/C)) / n_prime
    
    # Recursive calculation from 2nd farthest to nearest
    i_prime = n_prime
    for j in range(n-2, -1, -1):
        idx_i = sorted_dxy_idx[j]
        idx_i_plus_1 = sorted_dxy_idx[j+1]
        
        # Calculate i' = sum of weights up to current point
        i_prime -= w[j+1]
        
        # Compute the adjustment term
        adjustment = (1/K_prime) * ((min(K_prime, i_prime) * (n_prime-1) / i_prime) - K_prime)
        
        # Compute the difference in recursive formula
        term = (y_match[j] - y_match[j+1]) / (n_prime-1) * (harmonic_sum + adjustment)
        
        s[idx_i] = s[idx_i_plus_1] + term
        
    # Scale each value by their weights
    s[idx_n] = s[idx_n] * weight_n
    for j in range(n-2, -1, -1):
        idx_i = sorted_dxy_idx[j]
        s[idx_i] = s[idx_i] * w[j]

    return s 


def shapley_mc_single(D, z_test, K, kernel_fn, n_perms=1000):
    """
    Compute Shapley values for weighted KNN using Monte Carlo sampling.
    
    Args:
        D: List of tuples (x, y) where x is feature vector, y is label
        z_test: Test point tuple (x_test, y_test)
        K: Number of neighbors for KNN
        kernel_fn: Kernel function to compute weights
        n_perms: Number of permutations to sample
        
    Returns:
        Array of Shapley values for each data point
    """
    x_test, y_test = z_test
    n = len(D)
    if n == 0 or n == 1:
        return np.array([])
    
    # Calculate distances and corresponding features and labels
    dxy = [(distance(x, x_test), x, y) for x, y in D]
    
    # Store distances for each point
    distances = [d for d, _, _ in dxy]
    
    # Extract label matches (1 if label matches test point, 0 otherwise)
    y_match = [1 if y == y_test else 0 for _, _, y in dxy]
    
    # Compute weights for each point using the kernel function
    weights = [kernel_fn(d) for d in distances]
    
    # Compute number of classes
    unique_labels = set(y for _, y in D)
    C = len(unique_labels)
    
    # Initialize Shapley values array
    s = np.zeros(n)
    
    # Monte Carlo sampling - sample n_perms permutations
    for _ in range(n_perms):
        # Sample a random permutation
        perm = np.random.permutation(n)
        
        # For each position in the permutation, calculate marginal contribution
        subset = []  # Keep track of indices in the coalition so far
        utility_prev = 1 / C  # Default prediction without any points
        
        for pos, idx in enumerate(perm):
            # Add current index to the subset
            subset.append(idx)

            # Keep only the K nearest neighbors
            if len(subset) > K:
                # Sort subset by distance
                subset_dist = [(distances[i], i) for i in subset]
                sorted_subset = [i for _, i in sorted(subset_dist)]
                
                # Get K nearest neighbors from the subset
                knn_indices = sorted_subset[:K]
            else:
                knn_indices = subset
            
            # Calculate utility with the current coalition
            weighted_sum = sum(weights[i] * y_match[i] for i in knn_indices)
            weight_sum = sum(weights[i] for i in knn_indices)
            utility_curr = weighted_sum / weight_sum if weight_sum > 0 else 1 / C
            
            # Marginal contribution is the difference in utilities
            marginal = utility_curr - utility_prev
            
            # Add marginal contribution to Shapley value
            s[idx] += marginal
            
            # Update previous utility for next iteration
            utility_prev = utility_curr
    
    # Average over all permutations
    s /= n_perms
    
    return s
    