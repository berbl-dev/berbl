#!/usr/bin/env python3

import numpy as np  # type: ignore
from hypothesis import example, given  # type: ignore
import scipy.stats as sstats  # type: ignore
from hypothesis.strategies import floats, integers, lists, tuples  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs import *

D_X = 5
D_Y = 3
D_V = 5
N = 100
K = 10

X = arrays(
    np.float64,
    (N, D_X),
    # NOTE we do not test NaNs here and we restrict float width so we don't
    # get overflows when squaring
    elements=floats(allow_nan=False, allow_infinity=False, width=32))
Y = arrays(
    np.float64,
    (N, D_Y),
    # NOTE we do not test NaNs here and we restrict float width so we don't
    # get overflows when squaring
    elements=floats(allow_nan=False, allow_infinity=False, width=32))

positives = floats(
    min_value=0,
    # Have to exclude 0 because we otherwise divide by zero
    exclude_min=True,
    allow_nan=False,
    allow_infinity=False,
    width=32)

# NOTE we do not test NaNs here and we restrict float width so we don't
# get overflows when squaring
anyfloat = floats(allow_nan=False, allow_infinity=False, width=32)


@given(
    arrays(  # x
        np.float64, (D_X, ), elements=positives),
    arrays(  # y
        np.float64, (D_Y, ), elements=positives),
    arrays(  # M
        np.float64, (1, K), elements=positives),
    arrays(  # phi
        np.float64, (1, D_V), elements=positives),
    lists(  # W
        arrays(
            np.float64,
            (D_Y, D_X),
            # NOTE we do not test NaNs here and we restrict float width so we don't
            # get overflows when squaring
            elements=positives),
        min_size=K,
        max_size=K),
    lists(  # Lambda_1
        arrays(
            np.float64,
            (D_X, D_X),
            # NOTE we do not test NaNs here and we restrict float width so we don't
            # get overflows when squaring
            elements=positives),
        min_size=K,
        max_size=K),
    arrays(  # a_tau
        np.float64,
        (K, ),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=positives),
    arrays(  # b_tau
        np.float64,
        (K, ),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=positives),
    arrays(  # V
        np.float64,
        (D_V, K),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=positives),
)
def test_predictive_density(x, y, M, phi, W, Lambda_1, a_tau, b_tau, V):
    p = predictive_density(M, phi, W, Lambda_1, a_tau, b_tau, V)
    pxy = p(x)(y)

    pxy_ = 0
    g = mixing(M, phi, V)[0]
    for k in range(K):
        prod = 1
        for j in range(len(y)):
            prod *= sstats.t(loc=W[k][j] @ x,
                             scale=(1 + x @ (Lambda_1[k] @ x))**(-1) * a_tau[k]
                             / b_tau[k],
                             df=2 * a_tau[k]).pdf(y[j])
        pxy_ += g[k] * prod
    assert np.isclose(pxy, pxy_)


def test_matching_matrix():
    D_X = 5
    N = 100
    K = 10

    X = np.random.random((N, D_X))
    ranges = np.vstack([np.min(X, axis=0), np.max(X, axis=0)]).T
    ind = individual(ranges, K)

    assert matching_matrix(ind, X).shape == (N, K)


def test_train_classifier():
    D_X = 5
    D_Y = 3
    D_V = 5
    N = 100
    K = 10

    m_k = np.random.randint(low=0, high=2, size=N)
    X = np.random.random((N, D_X))
    Y = np.random.random((N, D_Y))

    W_k, Lambda_k_1, a_tau_k, b_tau_k, a_alpha_k, b_alpha_k = train_classifier(
        m_k, X, Y)


def test_train_mixing():
    D_X = 5
    D_Y = 3
    D_V = 5
    N = 100
    K = 10

    M = np.random.random((N, K))
    X = np.random.random((N, D_X))
    Y = np.random.random((N, D_Y))
    phi = np.random.random((N, D_V))
    W = [np.random.random((D_Y, D_X)) for i in range(K)]
    Lambda_1 = [np.random.random((D_X, D_X)) for i in range(K)]
    a_tau = np.random.random(K)
    b_tau = np.random.random(K)
    a_alpha = np.random.random(K)
    b_alpha = np.random.random(K)

    V, Lambda_V_1, a_beta, b_beta = train_mixing(M=M,
                                                 X=X,
                                                 Y=Y,
                                                 phi=phi,
                                                 W=W,
                                                 Lambda_1=Lambda_1,
                                                 a_tau=a_tau,
                                                 b_tau=b_tau)


def test_train_mix_weights():
    D_X = 5
    D_Y = 3
    D_V = 5
    N = 100
    K = 10

    M = np.random.random((N, K))
    X = np.random.random((N, D_X))
    Y = np.random.random((N, D_Y))
    W = [np.random.random((D_Y, D_X)) for i in range(K)]
    Lambda_1 = [np.random.random((D_X, D_X)) for i in range(K)]
    a_tau = np.random.random(K)
    b_tau = np.random.random(K)
    V = np.random.random((D_V, K))
    phi = np.random.random((N, D_V))
    a_beta = np.random.random(K)
    b_beta = np.random.random(K)

    V, Lambda_V_1 = train_mix_weights(M=M,
                                      X=X,
                                      Y=Y,
                                      phi=phi,
                                      W=W,
                                      Lambda_1=Lambda_1,
                                      a_tau=a_tau,
                                      b_tau=b_tau,
                                      V=V,
                                      a_beta=a_beta,
                                      b_beta=b_beta)


def test_hessian():
    D_V = 5
    N = 100
    K = 10

    phi = np.random.random((N, D_V))
    G = np.random.random((N, K))
    a_beta = np.random.random(K)
    b_beta = np.random.random(K)

    H = hessian(phi, G, a_beta, b_beta)
