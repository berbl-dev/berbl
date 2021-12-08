from copy import copy

import berbl.literal as literal
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from berbl.match.allmatch import AllMatch
from berbl.match.nomatch import NoMatch
from berbl.mixing_laplace import MixingLaplace
from berbl.rule import Rule
from berbl.utils import check_phi, matching_matrix
from hypothesis import given, settings  # type: ignore
from test_berbl import (Xs, assert_isclose, noshrinking, random_data,
                        random_states, rmatch1ds, ys)


@given(Xs(), ys(), random_states())
@settings(max_examples=50)
def test_no_match_no_weight(X, y, random_state):
    """
    Mixing a matching and a non-matching rule should result in 100/0 mixing
    weights.
    """
    cl1 = Rule(AllMatch()).fit(X, y)
    cl2 = Rule(NoMatch()).fit(X, y)

    mix = MixingLaplace(rules=[cl1, cl2], phi=None,
                        random_state=random_state).fit(X, y)
    G = mix.mixing(X)

    msg = (f"Mixing a not matching and a matching rule isn't correct\n"
           f"{G}\n"
           f"{mix.R_}\n"
           f"{mix.V_}")
    assert np.all(np.isclose(G, [1, 0])), msg


@given(st.lists(rmatch1ds(has_bias=True), min_size=9, max_size=10),
       random_data(N=100), random_states())
@settings(max_examples=50, deadline=None, phases=noshrinking)
def test_fit_like_literal(matchs, data, random_state):

    X, y = data

    rules = [Rule(match).fit(X, y) for match in matchs]

    phi = None
    mix = MixingLaplace(rules=rules, phi=phi,
                        random_state=copy(random_state)).fit(X, y)

    W = [rule.W_ for rule in rules]
    Lambda_1 = [rule.Lambda_1_ for rule in rules]
    a_tau = [rule.a_tau_ for rule in rules]
    b_tau = [rule.b_tau_ for rule in rules]

    M = matching_matrix(matchs, X)
    assert_isclose(M, np.hstack([rule.m_ for rule in rules]))

    Phi = check_phi(phi, X)
    V, Lambda_V_1, a_beta, b_beta = literal.train_mixing(
        M=M,
        X=X,
        Y=y,
        Phi=Phi,
        W=W,
        Lambda_1=Lambda_1,
        a_tau=a_tau,
        b_tau=b_tau,
        exp_min=mix.EXP_MIN,
        ln_max=mix.LN_MAX,
        random_state=copy(random_state))

    assert_isclose(mix.V_, V, label="V_")
    assert_isclose(mix.Lambda_V_1_, Lambda_V_1, label="Lambda_V_1_")
    assert_isclose(mix.a_beta_, a_beta, label="a_beta_")
    assert_isclose(mix.b_beta_, b_beta, label="b_beta_")
