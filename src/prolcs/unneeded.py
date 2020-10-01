#!/usr/bin/env python3

import numpy as np  # type: ignore


def drugo_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Actually, NumPy's * already does the job for us, it seems.

    (PDF p. 234)

    Given an a×b matrix or vector A, and c×d matrix or vector B, and a=c, b=d,
    drugo_prod(A, B) returns an a×b matrix that is the result of an element-wise
    multiplication of A and B.

    If a=c, d=1, that is, if B is a column vector with c elements, then every
    column of A is multiplied element-wise by B, and the result is returned. In
    NumPy terms, B.shape = (a).

    Analogously, if B is a row vector with b elements, then each row of A is
    multiplied element-wise by B, and the result is returned. In
    NumPy terms, B.shape = (1, b).
    """
    assert len(a.shape) == 2
    if a.shape == b.shape:
        return a * b
    # b is column vector, e.g. [[1], [2], [3]], b.shape = (-1, 1)
    elif len(b.shape) == 2 and a.shape[0] == b.shape[0] and b.shape[1] == 1:
        return a * b
    # b is row vector, e.g. [1, 2, 3]
    elif a.shape[1] == b.shape[0] and len(b.shape) == 1:
        return a * b
    else:
        raise ValueError(
            f"Incompatible arguments of shapes {a.shape}, {b.shape}")
