import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given, settings  # type: ignore
from prolcs.classifier import Classifier
from prolcs.match.allmatch import AllMatch
from prolcs.match.nomatch import NoMatch
from prolcs.mixing_laplace import MixingLaplace
from test_prolcs import Xs, rmatch1ds, ys


@given(rmatch1ds(), Xs(), ys())
@settings(max_examples=50)
def test_same_match_equal_weights(match, X, y):
    """
    Mixing two instances of the same classifier should result in 50/50 mixing
    weights.
    """
    cl = Classifier(match).fit(X, y)
    mix = MixingLaplace(classifiers=[cl, cl], phi=None).fit(X, y)
    G = mix.mixing(X)

    assert np.all(np.isclose(
        G, [0.5, 0.5])), (f"Mixing of the same classifiers isn't uniform"
                          f"{mix.R_}"
                          f"{mix.V_}")


@given(Xs(), ys())
@settings(max_examples=50)
def test_no_match_no_weight(X, y):
    """
    Mixing a matching and a non-matching classifier should result in 100/0
    mixing weights.
    """
    cl1 = Classifier(AllMatch()).fit(X, y)
    cl2 = Classifier(NoMatch()).fit(X, y)

    mix = MixingLaplace(classifiers=[cl1, cl2], phi=None).fit(X, y)
    G = mix.mixing(X)

    msg = (f"Mixing a not matching and a matching classifier isn't correct"
           f"{G}"
           f"{mix.R_}"
           f"{mix.V_}")
    assert np.all(np.isclose(G, [1, 0])), msg
