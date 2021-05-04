# import pytest  # type: ignore
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.match.radial1d_drugowitsch import RadialMatch1D
from prolcs.utils import add_bias

# TODO Reduce composite duplication from test_classifier


@st.composite
def match1d(draw, has_bias_column=True):
    a = draw(st.floats(min_value=0, max_value=100))
    b = draw(st.floats(min_value=0, max_value=50))
    return RadialMatch1D(a=a,
                         b=b,
                         ranges=np.array([[-1, 1]]),
                         has_bias_column=has_bias_column)


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
