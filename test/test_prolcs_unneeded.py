#!/usr/bin/env python3

import numpy as np  # type: ignore
from hypothesis import example, given  # type: ignore
from hypothesis.strategies import floats, integers, tuples  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.unneeded import *


@given(
    arrays(
        np.float64,
        tuples(integers(min_value=1, max_value=10),
               integers(min_value=1, max_value=10)),
        # NOTE we do not test NaNs here and we restrict float width so we don't
        # get overflows when squaring
        elements=floats(allow_nan=False, allow_infinity=False, width=32)))
@example(np.array([[0, 0]]))
def test_drugo_prod(x):
    a = x
    b = x
    r = drugo_prod(a, b)
    assert r.shape == a.shape
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            assert r[i][j] == a[i][j] * b[i][j]

    # a = [[1, 2], [3, 4]]
    # b = [[1], [2]] (column vector)
    a = x
    b = a[:, [0]]
    r = drugo_prod(a, b)
    assert r.shape == a.shape
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # NumPy indexes rows first, then columns
            assert r[i][j] == a[i][j] * b[i][0]

    # a = [[1, 2], [3, 4]]
    # b = [1, 2] (row vector)
    a = x
    b = a[0]
    r = drugo_prod(a, b)
    assert r.shape == a.shape
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # NumPy indexes rows first, then columns
            assert r[i][j] == a[i][j] * b[j]
