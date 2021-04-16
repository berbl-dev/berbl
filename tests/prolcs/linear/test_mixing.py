import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given, settings  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.allmatch import AllMatch
from prolcs.linear.classifier import Classifier
from prolcs.linear.mixing import Mixing
from prolcs.nomatch import NoMatch
from prolcs.radialmatch1d import RadialMatch1D
from prolcs.utils import add_bias


@st.composite
def match1ds(draw, has_bias_column=True):
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
               elements=st.floats(min_value=-1, max_value=1),
               unique=True))
    if bias_column:
        X = add_bias(X)
    return X


@st.composite
def ys(draw, N=10, D_y=1):
    return draw(
        arrays(np.float64, (N, D_y),
               elements=st.floats(min_value=-1000, max_value=1000)))


@given(match1ds(), Xs(), ys())
@settings(max_examples=50)
def test_same_match_equal_weights(match, X, y):
    """
    Mixing two instances of the same classifier should result in 50/50 mixing
    weights.
    """
    cl = Classifier(match).fit(X, y)
    mix = Mixing(classifiers=[cl, cl], phi=None).fit(X, y)
    G = mix.mixing(X)

    msg = (f"Mixing of the same classifiers isn't uniform"
           f"{mix.R_}"
           f"{mix.V_}")
    assert np.all(np.isclose(G, [0.5, 0.5])), msg


@given(Xs(), ys())
@settings(max_examples=50)
def test_no_match_no_weight(X, y):
    """
    Mixing a matching and a non-matching classifier should result in 100/0
    mixing weights.
    """
    cl1 = Classifier(AllMatch()).fit(X, y)
    cl2 = Classifier(NoMatch()).fit(X, y)

    mix = Mixing(classifiers=[cl1, cl2], phi=None).fit(X, y)
    G = mix.mixing(X)

    msg = (f"Mixing a not matching and a matching classifier isn't correct"
           f"{G}"
           f"{mix.R_}"
           f"{mix.V_}")
    assert np.all(np.isclose(G, [1, 0])), msg


# TODO train and check for similarity with mixing_laplace
