import pytest
import numpy as np
from functools import partial

from knnsvdup.helper import distance, kernel_value
from knnsvdup.unweighted import shapley


def test_shapley_dup():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    z_test = (np.array([0.0]), 1)
    sigma = 1
    kernel_fn = partial(kernel_value, sigma=sigma)
   
    shapley_values = shapley(D, z_test, K=1, kernel_fn=kernel_fn)
    answer = [4.30778595,  0.57500344, -6.73381352]

    assert np.allclose(shapley_values, answer, atol=1e-03)


def test_shapley_unweighted():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    z_test = (np.array([0.0]), 1)

    shapley_values = shapley(D, z_test, K=1)
    answer = [0.80555556,  0.30555556, -0.61111111]

    assert np.allclose(shapley_values, answer, atol=1e-03)
