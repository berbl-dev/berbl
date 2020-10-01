#!/usr/bin/env python3

import numpy as np  # type: ignore
from hypothesis import example, given, settings  # type: ignore
from hypothesis.strategies import floats, integers, tuples  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore


@given(
    arrays(
        np.float64,
        (100, 3),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(
        np.float64,
        (3, 3),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(
        np.float64,
        (3, ),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)))
@settings(deadline=None)
def test_vectorize(X, sigma, mu):
    mu_ = np.broadcast_to(mu, X.shape)
    delta = X - mu_
    M = np.sum(delta.T * (sigma @ delta.T), 0)

    for i in range(len(X)):
        m = (X[i] - mu) @ (sigma @ (X[i] - mu))
        assert np.isclose(m, M[i])


@given(
    arrays(
        np.float64,
        (100000, 3),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(
        np.float64,
        (3, 3),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(
        np.float64,
        (3, ),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)))
@settings(deadline=None)
def test_profile_vectorized(X, sigma, mu):
    mu_ = np.broadcast_to(mu, X.shape)
    delta = X - mu_
    M = np.sum(delta.T * (sigma @ delta.T), 0)


@given(
    arrays(
        np.float64,
        (100000, 3),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(
        np.float64,
        (3, 3),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(
        np.float64,
        (3, ),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)))
@settings(deadline=None)
def test_profile_loop(X, sigma, mu):
    M = np.repeat(False, (X.shape[0], ))
    for i in range(len(X)):
        M[i] = (X[i] - mu) @ (sigma @ (X[i] - mu)) > 0.5
