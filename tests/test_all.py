import pytest
import numpy as np
from functools import partial

from knnsvdup.helper import distance, kernel_value, approx_harmonic_sum, get_knn_acc
from knnsvdup.dup import shapley


def test_shapley_dup():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    z_test = (np.array([0.0]), 1)
    sigma = 1
    kernel_fn = partial(kernel_value, sigma=sigma)
   
    shapley_values = shapley(D, z_test, K=1, value_type="dup", kernel_fn=kernel_fn)
    answer = [ 4.30470185,  0.59768083, -4.40238268]

    assert np.allclose(shapley_values, answer, atol=1e-03)


def test_shapley_unweighted():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    z_test = (np.array([0.0]), 1)

    shapley_values = shapley(D, z_test, K=1, value_type="unweighted")
    answer = [0.80555556,  0.30555556, -0.61111111]

    assert np.allclose(shapley_values, answer, atol=1e-03)

    
def test_harmonic_sum():
    sums, sums_real = [], []
    real = 0
    for i in range(1, 1000000):
        sums.append(approx_harmonic_sum(i))
        real += 1/i
        sums_real.append(real)
    assert np.allclose(sums, sums_real, atol=1e-07)


def test_get_knn_acc():
    kernel_fn = lambda d: 1

    # Setup training data
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0, 0, 1, 1])
    
    # Setup validation data
    X_val = np.array([[0.5], [2.5]])
    y_val = np.array([0, 1])
    
    # Test with K=1
    acc_k1 = get_knn_acc(X_train, y_train, X_val, y_val, K=1, kernel_fn=kernel_fn)
    assert acc_k1 == 1.0  # Both test points should be classified correctly with K=1
    
    # Test with K=3
    acc_k3 = get_knn_acc(X_train, y_train, X_val, y_val, K=3, kernel_fn=kernel_fn)
    assert np.isclose(acc_k3, 0.666666667)  # Expected: 2/3
    
    # Test with empty training set
    acc_empty = get_knn_acc(np.array([]), np.array([]), X_val, y_val, K=1, kernel_fn=kernel_fn, C=2)
    assert acc_empty == 0.5  # With 2 unique classes, random accuracy should be 0.5


def test_shapley_mc_single():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    z_test = (np.array([0.0]), 1)
    sigma = 1
    kernel_fn = partial(kernel_value, sigma=sigma)
    
    # Set a fixed seed for reproducibility
    np.random.seed(42)
    
    shapley_values = shapley(D, z_test, K=1, value_type="mc", kernel_fn=kernel_fn, n_perms=1000)
    expected_approx = [ 0.6709,  0.1671, -0.338 ]
    assert np.allclose(shapley_values, expected_approx, atol=1e-01)
    
    # Also verify that increasing the number of permutations reduces variance
    np.random.seed(42)
    shapley_values_more_perms = shapley(D, z_test, K=1, value_type="mc", kernel_fn=kernel_fn, n_perms=5000)
    assert np.allclose(shapley_values_more_perms, expected_approx, atol=1e-03)

    shapley_values_bf = shapley(D, z_test, K=1, value_type="bf")
    assert np.allclose(shapley_values_more_perms, shapley_values_bf, atol=0.03)
