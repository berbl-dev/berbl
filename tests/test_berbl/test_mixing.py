import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given, settings  # type: ignore
from berbl.rule import Rule
from berbl.match.allmatch import AllMatch
from berbl.match.nomatch import NoMatch
from berbl.mixing import Mixing
from test_berbl import Xs, rmatch1ds, ys

# TODO Test using n-d inputs (e.g. radial instead of radial1d)
# TODO train and check for similarity with mixing_laplace


@given(rmatch1ds(), Xs(), ys())
@settings(max_examples=50)
def test_same_match_equal_weights(match, X, y):
    """
    Mixing two instances of the same rule should result in 50/50 mixing weights.
    """
    cl = Rule(match).fit(X, y)
    mix = Mixing(rules=[cl, cl], phi=None).fit(X, y)
    G = mix.mixing(X)

    msg = (f"Mixing of the same rules isn't uniform"
           f"{mix.R_}"
           f"{mix.V_}")
    # We ever so slightly weaken atol (default value is 1e-8).
    assert np.all(np.isclose(G, [0.5, 0.5], atol=1e-6)), msg


@given(Xs(), ys())
@settings(max_examples=50)
def test_no_match_no_weight(X, y):
    """
    Mixing a matching and a non-matching rule should result in 100/0 mixing
    weights.
    """
    cl1 = Rule(AllMatch()).fit(X, y)
    cl2 = Rule(NoMatch()).fit(X, y)

    mix = Mixing(rules=[cl1, cl2], phi=None).fit(X, y)
    G = mix.mixing(X)

    msg = (f"Mixing a not matching and a matching rule isn't correct"
           f"{G}"
           f"{mix.R_}"
           f"{mix.V_}")
    assert np.all(np.isclose(G, [1, 0])), msg
