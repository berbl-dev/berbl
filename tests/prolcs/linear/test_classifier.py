# import pytest  # type: ignore
from hypothesis import given, settings  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from prolcs.linear.classifier import Classifier
from prolcs.radialmatch1d import RadialMatch1D
from prolcs.utils import add_bias


@st.composite
def match1d(draw, has_bias_column=True):
    a = draw(st.floats(min_value=0, max_value=100))
    b = draw(st.floats(min_value=0, max_value=50))
    return RadialMatch1D(a=a,
                         b=b,
                         ranges=(-1, 1),
                         has_bias_column=has_bias_column)


@st.composite
def Xs(draw, N=10, D_X=1, bias_column=True):
    X = draw(
        arrays(np.float64, (N, D_X),
               elements=st.floats(min_value=-1, max_value=1)))
    if bias_column:
        X = add_bias(X)
    return X


@st.composite
def ys(draw, N=10, D_y=1):
    return draw(arrays(np.float64, (N, D_y),
                elements=st.floats(min_value=-1000, max_value=1000)))


@given(match1d(), Xs(), ys())
@settings(deadline=None, max_examples=15)
def test_fit_inc_L_q(match, X, y):
    """
    “Each parameter update either increases L_q or leaves it unchanged (…). If
    this is not the case, then the implementation is faulty and/or suffers from
    numerical instabilities.” [PDF p. 237]

    assert delta_L_q >= 0, f"delta_L_q = {delta_L_q} < 0"
    """
    max_iters = range(0, 100, 10)
    L_qs = np.array([None] * len(max_iters))
    for i in range(len(max_iters)):
        L_qs[i] = Classifier(match, MAX_ITER=max_iters[i]).fit(X, y).L_q

    assert np.all(np.diff(L_qs) >= 0)
