import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from berbl.match.hardinterval import HardInterval, mirror
from hypothesis import given  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from test_berbl import Xs_and_matchs, himatchs, random_states


@given(Xs_and_matchs(himatchs))
def test_match_never_nana(X_and_match):
    X, match = X_and_match
    assert np.all(~np.isnan(match.match(X)))


@given(Xs_and_matchs(himatchs))
def test_match_prob_bounds(X_and_match):
    X, match = X_and_match
    m = match.match(X)
    # All matching functions should match all samples, at least a little bit.
    assert np.all(0 < m)
    assert np.all(m <= 1)


@given(Xs_and_matchs(himatchs))
def test_match_respect_genotype_space(X_and_match):
    X, match = X_and_match
    assert np.all((0 <= match.center) & (match.center <= match.res_center))
    assert np.all((0 <= match.spread) & (match.spread <= match.res_spread))


def test_match_min_max_values():
    match = HardInterval(center=np.array([0]), spread=np.array([0]))

    assert match.center_phen == np.array(match.x_min)
    assert match.center_phen == np.array(match.x_min)

    res_center = 2**8
    res_spread = 2 * 2**8
    match = HardInterval(center=np.array([res_center]),
                         spread=np.array([res_spread]),
                         res_center=res_center,
                         res_spread=res_spread)

    assert match.spread_phen == np.array([match.x_max])
    assert match.spread_phen == np.array([match.x_max])


@given(himatchs(), random_states())
def test_match_mutation_bounds(match, random_state):

    match2 = match.mutate(random_state)

    assert np.all((0 <= match2.center) & (match2.center <= match2.res_center))
    assert np.all((0 <= match2.spread) & (match2.spread <= match2.res_spread))

    # Upper and lower bounds are not in [x_min, x_max] but may be in [x_min -
    # (x_max - x_min) / 2, x_max + (x_max - x_min) / 2] because we don't want to
    # introduce epistasis into the genotype (we'd need to make the domain of
    # center depend on the current value of spread in order to ensure [x_min,
    # x_max] are respected).
    assert np.all(match.x_min - (match.x_max - match.x_min) / 2 <= match2.l)
    assert np.all(match2.u <= match.x_max + (match.x_max - match.x_min) / 2)


@given(arrays(np.float64, (10,), elements=st.floats(min_value=-1, max_value=1)))
def test_mirror(a):
    a_ = mirror(a, a_min=-0.75, a_max=0.75)
    assert np.all(-0.75 <= a_)
    assert np.all(a_ <= 0.75)

    a_ = mirror(a, a_min=-0.75, a_max=0.75, exclude_min=True, exclude_max=True)
    assert np.all(-0.75 < a_)
    assert np.all(a_ < 0.75)
