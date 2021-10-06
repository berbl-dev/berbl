# import pytest  # type: ignore
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.match.softinterval1d_drugowitsch import SoftInterval1D
from prolcs.utils import add_bias

# TODO Reduce composite duplication from test_classifier


@st.composite
def match1d(draw, has_bias=True):
    l_ = draw(st.floats(min_value=-1, max_value=1))
    u_ = draw(st.floats(min_value=-1, max_value=1).filter(lambda u_: u_ != l_))
    l = min(l_, u_)
    u = max(l_, u_)
    return SoftInterval1D(l=l,
                          u=u,
                          has_bias=has_bias)


@st.composite
def Xs(draw, N=10, D_X=1, bias_column=True):
    X = draw(
        arrays(np.float64, (N, D_X),
               elements=st.floats(min_value=-1, max_value=1)))
    if bias_column:
        X = add_bias(X)
    return X


@given(match1d(), Xs())
def test_match_never_nan(match, X):
    assert np.all(~np.isnan(match.match(X)))


@given(match1d(), Xs())
def test_match_prob_bounds(match, X):
    m = match.match(X)
    # All matching functions should match all samples, at least a little bit.
    assert np.all(0 < m)
    assert np.all(m <= 1)
