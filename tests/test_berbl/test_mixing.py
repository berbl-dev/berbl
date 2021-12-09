import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from berbl.match.allmatch import AllMatch
from berbl.match.nomatch import NoMatch
from berbl.mixing import Mixing
from berbl.rule import Rule
from hypothesis import given, settings  # type: ignore
from test_berbl import Xs, random_states, rmatch1ds, ys

# TODO Test using n-d inputs (e.g. radial instead of radial1d)


@given(Xs(), ys(), random_states())
@settings(max_examples=50)
def test_no_match_no_weight(X, y, random_state):
    """
    Mixing a matching and a non-matching rule should result in 100/0 mixing
    weights.
    """
    cl1 = Rule(AllMatch()).fit(X, y)
    cl2 = Rule(NoMatch()).fit(X, y)

    mix = Mixing(rules=[cl1, cl2], phi=None,
                 random_state=random_state).fit(X, y)
    G = mix.mixing(X)

    msg = (f"Mixing a not matching and a matching rule isn't correct"
           f"{G}"
           f"{mix.R_}"
           f"{mix.V_}")
    assert np.all(np.isclose(G, [1, 0])), msg
