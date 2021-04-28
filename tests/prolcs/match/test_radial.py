# import pytest  # type: ignore
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.match.radial import RadialMatch
from prolcs.utils import add_bias, get_ranges


@st.composite
def dimensions(draw):
    return draw(st.integers(min_value=2, max_value=50))


@st.composite
def seeds(draw):
    # Highest possible seed is `2**32 - 1` for NumPy legacy generators.
    return draw(st.integers(min_value=0, max_value=2**32 - 1))


@st.composite
def Xs(draw, N=10, D_X=1, bias_column=True, unique=True):
    X = draw(
        arrays(np.float64, (N, D_X),
               elements=st.floats(min_value=-1, max_value=1),
               unique=unique))
    if bias_column:
        X = add_bias(X)
    return X


@st.composite
def dims_and_Xs_and_matchs(draw, N=10, bias_column=True, unique=True):
    D_X = draw(dimensions())
    X = draw(Xs(N=N, D_X=D_X, bias_column=bias_column, unique=unique))
    ranges = get_ranges(X)
    if bias_column:
        ranges = ranges[1:]
    seed = draw(seeds())
    match = RadialMatch.random(ranges=ranges, random_state=seed)
    return D_X, X, match


@st.composite
def dims_and_epsilons(draw):
    D_X = draw(dimensions())
    epsilon = draw(
        arrays(np.float64, (D_X, ),
               elements=st.floats(min_value=1e-5, max_value=1)))
    return D_X, epsilon


@given(dims_and_Xs_and_matchs())
def test_match_never_nan(dXm):
    d, X, match = dXm
    assert np.all(~np.isnan(match.match(X)))


@given(dims_and_Xs_and_matchs())
def test_match_prob_bounds(dXm):
    d, X, match = dXm
    m = match.match(X)
    assert np.all(0 < m)
    assert np.all(m <= 1)


@given(dimensions(), seeds())
def test_random_eigvals_gt_0(D_X, seed):
    ranges = np.repeat([[-1, 1]], D_X, axis=0)
    rmatch = RadialMatch.random(ranges=ranges,
                                has_bias=False,
                                random_state=seed)
    assert np.all(rmatch.eigvals > 0), f"Eigenvalues not > 0: {eigvals}"


@given(dims_and_epsilons(), seeds())
def test_match_mu_is_mode(dim_and_epsilon, seed):
    """
    We check whether matching value at `x = rmatch.mu` is larger than matching
    values a short distance away.

    Note that we currently only check larger/lower values on the diagonal ``f(x)
    = x`` since we use the same epsilon value in each dimension.
    """
    D_X, epsilon = dim_and_epsilon
    ranges = np.repeat([[-1, 1]], D_X, axis=0)
    rmatch = RadialMatch.random(ranges=ranges, random_state=seed)
    # TODO Consider using check_array in match, add_bias
    X = add_bias(np.array([rmatch.mu]))
    m = rmatch.match(X)
    Xe1 = add_bias(np.array([rmatch.mu]) - epsilon)
    me1 = rmatch.match(Xe1)
    Xe2 = add_bias(np.array([rmatch.mu]) + epsilon)
    me2 = rmatch.match(Xe2)
    assert np.all(
        me1 < m), f"Distribution mode is not at mu: {m[0]} < {me1[0]}"
    assert np.all(
        me2 < m), f"Distribution mode is not at mu: {m[0]} < {me2[0]}"


@given(dimensions(), seeds())
def test_match_not_at_mu(D_X, seed):
    ranges = np.repeat([[-1, 1]], D_X, axis=0)
    rmatch = RadialMatch.random(ranges=ranges, random_state=seed)
    X = add_bias(np.array([rmatch.mu - 1e-2]))
    m = rmatch.match(X)
    assert np.all(m <= 1), f"Does match with >= 100% at non-mode point: {m}"


@given(dimensions(), seeds())
def test_match_symmetric_covariance(D_X, seed):
    ranges = np.repeat([[-1, 1]], D_X, axis=0)
    rmatch = RadialMatch.random(ranges=ranges, random_state=seed)
    cov = rmatch._covariance()
    assert np.allclose(cov, cov.T), f"Covariance matrix is not symmetrical"


# TODO Test whether we match >80% of uniformly distributed samples using random
# init (when random init has been implemented).
