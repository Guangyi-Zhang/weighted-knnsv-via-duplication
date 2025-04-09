import pytest
import numpy as np
from functools import partial

from knnsvdup.helper import distance, kernel_value
from knnsvdup.unweighted import shapley_unweighted_bf


def test_shapley_unweighted():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    z_test = (np.array([0.0]), 1)

    shapley_values = shapley_unweighted_bf(D, z_test, K=1)
    answer = [ 0.83333333,  0.33333333, -0.16666667]

    assert np.allclose(shapley_values, answer, atol=1e-03)
