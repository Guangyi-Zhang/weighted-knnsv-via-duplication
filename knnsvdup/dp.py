import numpy as np
import math
import time
from scipy.special import comb
from tqdm import tqdm

from knnsvdup.helper import distance

# Code adapted from https://github.com/Jiachen-T-Wang/weighted-knn-shapley/tree/main

"""
Example:

import numpy as np
from sklearn.model_selection import train_test_split

from exps.prepare_data import get_processed_data
from knnsvdup.dp import fastweighted_knn_shapley


X, y = get_processed_data("click")
X = X[:100]
y = y[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalize the data
X_mean, X_std = np.mean(X_train, 0), np.std(X_train, 0)
normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
X_train, X_test = normalizer_fn(X_train), normalizer_fn(X_test)

print(f"X_train.shape, X_test.shape: {X_train.shape}, {X_test.shape}")

eps = 0
K = 5
dis_metric = 'l2'
kernel = 'rbf'
debug = False
n_bits = 3
temp = 0.1

sv, sv_lst = fastweighted_knn_shapley(X_train, y_train, X_test, y_test, eps=eps, K=K, dis_metric=dis_metric, 
                                      kernel=kernel, debug=True, n_bits=n_bits, collect_sv=True, temp=temp)
"""


def normalize_weight(weight, method="dividemax"):
    if method == "zeroone":
        if max(weight) - min(weight) > 0:
            weight = (weight - min(weight)) / (max(weight) - min(weight))
    elif method == "dividemax":
        weight = weight / max(weight)
    return weight


def compute_dist(x_train_few, x_test, dis_metric):
    if dis_metric == "cosine":
        distance = -np.dot(x_train_few, x_test) / np.linalg.norm(x_train_few, axis=1)
    else:
        distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
    return distance


# temp: temperature coefficient
def compute_weights(distance, kernel, temp=0.9):
    distance /= max(distance)

    if kernel == "rbf":
        weight = np.exp(-(distance) / temp)
    elif kernel == "plain":
        weight = -distance
    elif kernel == "uniform":
        weight = np.ones(len(distance))
    else:
        exit(1)

    return weight


def adjust_weights(weight, y_train_few, y_test, y_consider):
    # Sanity check
    assert y_consider >= 0
    assert y_consider != y_test

    adjusted_weights = np.zeros_like(weight)
    adjusted_weights[y_train_few == y_test] = weight[y_train_few == y_test]
    adjusted_weights[y_train_few == y_consider] = -weight[y_train_few == y_consider]
    return adjusted_weights.astype(np.float64)


def prepare_weights(
    x_train_few,
    y_train_few,
    x_test,
    y_test,
    dis_metric="cosine",
    kernel="rbf",
    y_consider=None,
    temp=1,
):
    C = max(y_train_few) + 1

    # Compute distance
    distance = compute_dist(x_train_few, x_test, dis_metric)

    # Compute weights
    weight = compute_weights(distance, kernel, temp=temp)

    # We normalize each weight to [0, 1]
    weight = normalize_weight(weight)

    # Adjust weights to give sign
    if C == 2:
        weight = weight * (2 * (y_train_few == y_test) - 1)
    else:
        weight = adjust_weights(weight, y_train_few, y_test, y_consider)

    return weight, distance


def quantize(value, n_bits):
    """
    Discretizes a real number (or a numpy array of real numbers) between 0 and 1 into n_bits
    and returns its integer representation.

    :param value: Real number or numpy array of real numbers to be quantized (between 0 and 1)
    :param n_bits: Number of bits for discretization
    :return: Integer representation of the quantized value
    """
    n_values = 2**n_bits
    quantized_value = np.round(value * (n_values - 1))

    return quantized_value.astype(int)


def quantize_to_real(value, n_bits):
    """
    Discretizes a real number (or a numpy array of real numbers) between 0 and 1 into n_bits
    and returns its discretized real number representation.

    :param value: Real number or numpy array of real numbers to be quantized (between 0 and 1)
    :param n_bits: Number of bits for discretization
    :return: Real number representation of the quantized value
    """
    n_values = 2**n_bits
    quantized_value = np.round(value * (n_values - 1))

    # Convert the quantized value back to the range [0, 1]
    discretized_real_value = quantized_value / (n_values - 1)

    return discretized_real_value


def get_range_binary(weight_disc, K):
    sort_pos = np.sort(weight_disc[weight_disc > 0])[::-1]
    sort_neg = np.sort(weight_disc[weight_disc < 0])

    weight_max_disc = round(
        sum(sort_pos[: min(K, len(sort_pos))])
    )  # sum of top K positive weights
    weight_min_disc = round(sum(sort_neg[: min(K, len(sort_neg))]))

    # Use range to get all integers between weight_min_disc and weight_max_disc inclusive
    all_possible = np.array(list(range(weight_min_disc, weight_max_disc + 1)))

    N_possible = len(all_possible)

    return weight_max_disc, weight_min_disc, N_possible, all_possible


# Find which endpoint x is closer to
def closest_endpoint(x, weight_min_disc, weight_max_disc):
    if weight_min_disc <= x <= weight_max_disc:
        return x

    if abs(x - weight_min_disc) < abs(x - weight_max_disc):
        return weight_min_disc
    else:
        return weight_max_disc


# eps: precision
# n_bits: number of bit representation
def fastweighted_knn_shapley_binary_single_changebase(
    weights, distance, K, eps=0, debug=True, n_bits=3
):
    if debug:
        print("Original Weights={}".format(weights))

    N = len(distance)
    sv = np.zeros(N)

    # Weights Discretization
    # Note: weight_disc are integers
    weight_disc = quantize(weights, n_bits)

    # reorder weight_disc based on distance rank
    rank = np.argsort(distance)
    weight_disc = weight_disc[rank]

    # maximum possible range of weights
    weight_max_disc, weight_min_disc, V, all_possible = get_range_binary(weight_disc, K)

    if debug:
        print("weight_disc", weight_disc)
        print("weight_max_disc", weight_max_disc)
        print("weight_min_disc", weight_min_disc)
        print("all_possible", all_possible)
        print("weight_disc", weight_disc)

    val_ind_map = {}
    for j, val in enumerate(all_possible):
        val_ind_map[val] = j

    index_zero = val_ind_map[0]

    if debug:
        print(
            "index of zero: {}, value check = {}".format(
                index_zero, all_possible[index_zero]
            )
        )

    # error bound; TODO: improve the efficiency
    def E(mstar):
        mstar += 1
        assert mstar >= K
        A = np.sum([1 / (m - K) - 1 / m for m in range(mstar + 1, N + 1)])
        B = (
            np.sum(
                [(comb(N, l) - comb(mstar, l)) / comb(N - 1, l) for l in range(1, K)]
            )
            / N
        )
        return A + B

    t_Ei = time.time()

    # Compute the smallest m_star s.t. E(m_star) <= eps
    err = 0
    if eps > 0 and N > K:
        m_star = N - 1
        while err < eps and m_star + 1 >= K:
            m_star -= 1
            err = E(m_star)
    else:
        m_star = N - 1

    t_Ei = time.time() - t_Ei

    print("m_star = {}".format(m_star))

    t_F = time.time()

    # set size+1 for each entry for convenience
    F = np.zeros((m_star + 1, N + 1, V))

    # Initialize for l=1
    for m in range(0, m_star + 1):
        wm = weight_disc[m]
        ind_m = val_ind_map[wm]
        F[m, 1, ind_m] = 1

    # For 2 <= l <= K-1
    for l in range(2, K):
        for m in range(l - 1, m_star + 1):
            wm = weight_disc[m]
            check_vals = all_possible - wm

            for j, s in enumerate(all_possible):
                check_val = check_vals[j]
                if check_val < weight_min_disc or check_val > weight_max_disc:
                    F[m, l, j] = 0
                else:
                    ind_sm = val_ind_map[check_val]
                    F[m, l, j] += np.sum(F[:m, l - 1, ind_sm])

    if debug:
        print("Computed F; Time: {}".format(time.time() - t_F))

    t_smallloop = 0
    t_computeGi = 0

    I = np.array([comb(N - 1, l) for l in range(0, N)])

    comb_values = np.zeros(N)
    comb_values[K:] = 1.0 / np.array([comb(m, K) for m in range(K, N)])
    deno_values = np.arange(0, N) + 1

    t_Fi = time.time()

    Fi = np.zeros((m_star + 1, N, V))

    t_Fi_largerloop = 0

    sv_cache = {}
    sv_cache[0] = 0

    for i in range(N):
        wi = weight_disc[i]

        if wi in sv_cache:
            sv[i] = sv_cache[wi]

        else:
            t_Fi_start = time.time()

            # set size+1 for each entry for convenience
            Fi[:, :, :] = 0

            # Initialize for l=1
            Fi[:, 1, :] = F[:, 1, :]

            if i <= m_star:
                Fi[i, 1, :] = 0

            check_vals = all_possible - wi
            valid_indices = np.logical_and(
                check_vals >= weight_min_disc, check_vals <= weight_max_disc
            )
            invalid_indices = ~valid_indices
            mapped_inds = np.array(
                [val_ind_map[val] for val in check_vals[valid_indices]]
            )

            # For 2 <= l <= K-1
            for l in range(2, K):
                Fi[l - 1 : i, l, :] = F[l - 1 : i, l, :]

            for l in range(2, K):
                Fi[max(l - 1, i + 1) : (m_star + 1), l, valid_indices] = (
                    F[max(l - 1, i + 1) : (m_star + 1), l, valid_indices]
                    - Fi[max(l - 1, i + 1) : (m_star + 1), l - 1, mapped_inds]
                )
                Fi[max(l - 1, i + 1) : (m_star + 1), l, invalid_indices] = F[
                    max(l - 1, i + 1) : (m_star + 1), l, invalid_indices
                ]

            # if debug:
            #   t_smallloop += time.time()-t_Fi_start
            #   print('i={}, small_loop={}'.format(i, time.time()-t_Fi_start))

            t_Gi = time.time()

            Gi = np.zeros(N)

            if wi > 0:
                start_val, end_val = (
                    max(-wi + 1, weight_min_disc),
                    closest_endpoint(0, weight_min_disc, weight_max_disc),
                )
                start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]
            elif wi < 0:
                start_val, end_val = (
                    closest_endpoint(1, weight_min_disc, weight_max_disc),
                    min(-wi, weight_max_disc),
                )
                start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]
            else:
                exit(1)

            for m in range(m_star + 1):
                if i != m:
                    Gi[1:K] += np.sum(Fi[m, 1:K, start_ind : end_ind + 1], axis=1)

            # Precompute Ri
            Ri = np.zeros(N)

            if wi > 0:
                start_val, end_val = max(-wi + 1, weight_min_disc), wi
            elif wi < 0:
                start_val, end_val = wi + 1, min(-wi, weight_max_disc)
            start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]

            R0 = np.sum(Fi[: max(i + 1, K), K - 1, start_ind : end_ind + 1], axis=0)

            for m in range(max(i + 1, K), m_star + 1):
                wm = weight_disc[m]
                if wi > 0 and wm < wi:
                    end_val = max(-wm, weight_min_disc)
                    end_val = min(end_val, weight_max_disc)
                    end_ind_m = val_ind_map[end_val]
                    Ri[m] = np.sum(R0[: end_ind_m + 1 - start_ind])
                elif wi < 0 and wm > wi:
                    start_val = max(-wm + 1, weight_min_disc)
                    start_val = min(start_val, weight_max_disc)
                    start_ind_m = val_ind_map[start_val]
                    Ri[m] = np.sum(R0[start_ind_m - start_ind :])

                R0 += Fi[m, K - 1, start_ind : end_ind + 1]

            if debug:
                print("Ri={}".format(Ri))

            sv[i] = np.sum(
                Ri[max(i + 1, K) :]
                * comb_values[max(i + 1, K) :]
                * N
                / deno_values[max(i + 1, K) :]
            )
            sv[i] += np.sum(Gi[1:K] / I[1:K])

            if wi < 0:
                sv[i] = -sv[i]
            elif wi > 0:
                sv[i] += 1  # for l=0

            t_computeGi += time.time() - t_Gi

            t_Fi_largerloop += time.time() - t_Fi_start

    print(
        "Computed Fi; Time: {}, Ei_time={}, SmallLoop={}, t_computeGi={}, t_Fi_largerloop={}".format(
            time.time() - t_Fi, t_Ei, t_smallloop, t_computeGi, t_Fi_largerloop
        )
    )

    weight_disc_real = quantize_to_real(weights, n_bits)

    sv_real = np.zeros(N)
    sv_real[rank] = sv
    sv = sv_real

    print(
        "Sanity check: sum of SV = {}, U(N)-U(empty)={}".format(
            np.sum(sv) / N, int(np.sum(weight_disc[:K]) > 0)
        )
    )

    sv = sv / N

    if debug:
        print("weights (discretized):", weight_disc_real)
        print(sv)

    return sv


def fastweighted_knn_shapley(
    x_train_few,
    y_train_few,
    x_val_few,
    y_val_few,
    K,
    eps,
    dis_metric="cosine",
    kernel="rbf",
    debug=False,
    n_bits=3,
    collect_sv=False,
    temp=1,
):
    # Currently only work for K>1
    assert K > 1

    N = len(y_train_few)
    sv = np.zeros(N)

    n_test = len(y_val_few)
    C = max(y_train_few) + 1

    print("Number of classes = {}".format(C))

    distinct_classes = np.arange(C)

    sv_lst = []

    for i in tqdm(range(n_test)):
        x_test, y_test = x_val_few[i], y_val_few[i]

        if C == 2:
            weight, distance = prepare_weights(
                x_train_few,
                y_train_few,
                x_test,
                y_test,
                dis_metric,
                kernel,
                y_consider=None,
                temp=temp,
            )
            sv_i = fastweighted_knn_shapley_binary_single_changebase(
                weight, distance, K, eps, debug, n_bits
            )
        else:
            sv_i = np.zeros(N)
            classes_to_enumerate = distinct_classes[distinct_classes != y_test]
            for c in classes_to_enumerate:
                weight, distance = prepare_weights(
                    x_train_few,
                    y_train_few,
                    x_test,
                    y_test,
                    dis_metric,
                    kernel,
                    y_consider=c,
                    temp=temp,
                )
                nonzero_ind = np.nonzero(weight)[0]
                sv_temp = fastweighted_knn_shapley_binary_single_changebase(
                    weight[nonzero_ind],
                    distance[nonzero_ind],
                    K=min(K, len(nonzero_ind)),
                    eps=eps,
                    debug=debug,
                    n_bits=n_bits,
                )
                sv_i[nonzero_ind] += sv_temp

        sv += sv_i
        sv_lst.append(sv_i)

    if collect_sv:
        return sv, sv_lst
    else:
        return sv
