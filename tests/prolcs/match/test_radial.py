# import pytest  # type: ignore
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
import prolcs.match.radial as radial
import scipy.stats as sst
from hypothesis import given, settings  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.match.radial import RadialMatch, _rotate, mutate
from prolcs.utils import add_bias, radius_for_ci, space_vol
from sklearn.utils import check_random_state  # type: ignore


@st.composite
def dimensions(draw):
    return draw(st.integers(min_value=2, max_value=50))


@st.composite
def seeds(draw):
    # Highest possible seed is `2**32 - 1` for NumPy legacy generators.
    return draw(st.integers(min_value=0, max_value=2**32 - 1))


@st.composite
def Xs(draw, N=10, dX=1, has_bias=True, unique=True):
    """
    Input values normalized to ``[-1, 1]^dX``.
    """
    X = draw(
        arrays(np.float64, (N, dX),
               elements=st.floats(min_value=-1, max_value=1),
               unique=unique))
    if has_bias:
        X = add_bias(X)
    return X


@st.composite
def dims_and_Xs_and_matchs(draw, N=10, has_bias=True, unique=True):
    dX = draw(dimensions())
    X = draw(Xs(N=N, dX=dX, has_bias=has_bias, unique=unique))
    seed = draw(seeds())
    match = RadialMatch.random_ball(dX=dX,
                                    has_bias=has_bias,
                                    random_state=seed)
    return dX, X, match


@st.composite
def dims_and_epsilons(draw):
    dX = draw(dimensions())
    epsilon = draw(
        arrays(np.float64, (dX, ),
               elements=st.floats(min_value=1e-5, max_value=1)))
    return dX, epsilon


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


@given(dimensions(), st.booleans(), seeds())
def test_random_eigvals_gt_0(dX, has_bias, seed):
    rmatch = RadialMatch.random_ball(dX=dX,
                                     has_bias=has_bias,
                                     random_state=seed)
    assert np.all(rmatch.eigvals > 0), f"Eigenvalues not > 0: {eigvals}"


@given(dims_and_epsilons(), seeds())
@settings(deadline=None)
def test_match_mean_is_mode(dim_and_epsilon, seed):
    """
    We check whether matching value at `x = rmatch.mean` is larger than matching
    values a short distance away.

    Note that we currently only check larger/lower values on the diagonal ``f(x)
    = x`` since we use the same epsilon value in each dimension.
    """
    dX, epsilon = dim_and_epsilon
    # +1 due to has_bias=True.
    rmatch = RadialMatch.random_ball(dX=dX, has_bias=True, random_state=seed)
    # TODO Consider using check_array in match, add_bias
    X = add_bias(np.array([rmatch.mean]))
    m = rmatch.match(X)
    Xe1 = add_bias(np.array([rmatch.mean]) - epsilon)
    me1 = rmatch.match(Xe1)
    Xe2 = add_bias(np.array([rmatch.mean]) + epsilon)
    me2 = rmatch.match(Xe2)
    assert np.all(
        me1 < m), f"Distribution mode is not at mean: {m[0]} < {me1[0]}"
    assert np.all(
        me2 < m), f"Distribution mode is not at mean: {m[0]} < {me2[0]}"


@given(dimensions(), st.booleans(), seeds())
def test_match_not_at_mean(dX, has_bias, seed):
    rmatch = RadialMatch.random_ball(dX=dX,
                                     has_bias=has_bias,
                                     random_state=seed)
    if has_bias:
        X = add_bias(np.array([rmatch.mean - 1e-2]))
    else:
        X = np.array([rmatch.mean - 1e-2])
    m = rmatch.match(X)
    assert np.all(m <= 1), f"Does match with >= 100% at non-mode point: {m}"


@given(dimensions(), st.booleans(), seeds())
def test_match_symmetric_covariance(dX, has_bias, seed):
    rmatch = RadialMatch.random_ball(dX=dX,
                                     has_bias=has_bias,
                                     random_state=seed)
    cov = rmatch._covariance()
    assert np.allclose(cov, cov.T), f"Covariance matrix is not symmetrical"


@given(dimensions(), st.booleans(), st.floats(min_value=0.01, max_value=0.99),
       seeds())
def test_match_mutate_positive_definite(dX, has_bias, volstdfactor, seed):
    """
    A radial basis function–based matching function's covariance matrix has to
    stay positive definite under mutation.
    """
    rmatch = RadialMatch.random_ball(dX=dX,
                                     has_bias=has_bias,
                                     cover_confidence=0.5,
                                     random_state=seed)
    rmatch_ = radial.mutate(rmatch,
                            volstd=volstdfactor
                            * rmatch.covered_vol(cover_confidence=0.5),
                            cover_confidence=0.5,
                            random_state=seed)
    cov = rmatch_._covariance()
    assert np.all(np.linalg.eigvals(cov) > 0
                  ), f"Covariance matrix not positive definite after mutation"


@st.composite
def dims_and_idx_and_Xs(draw, N=10, unique=True):
    dX = draw(dimensions())
    i1 = draw(st.integers(min_value=0, max_value=dX - 1))
    i2 = draw(
        st.integers(min_value=0, max_value=dX - 1).filter(lambda i: i != i1))
    X = draw(Xs(N=N, dX=dX, has_bias=False, unique=unique))
    return dX, i1, i2, X


@given(dims_and_idx_and_Xs(), seeds())
@settings(deadline=None)
def test_rotate_eigvecs_180(diix, seed):
    """
    Rotating by 180° doesn't change radial-basis function–based match functions.
    """
    dX, i1, i2, X = diix
    rmatch = RadialMatch.random_ball(dX=dX, has_bias=False, random_state=seed)
    eigvecs_ = _rotate(rmatch.eigvecs, 180, i1, i2)
    rmatch_ = RadialMatch(mean=rmatch.mean,
                          has_bias=False,
                          eigvals=rmatch.eigvals,
                          eigvecs=eigvecs_)
    assert np.allclose(rmatch.match(X), rmatch_.match(X))


@given(dimensions(), seeds(), st.booleans())
def test_eigvals_same_order_as_eigvecs(dX, has_bias, seed):
    rmatch = RadialMatch.random_ball(dX=dX,
                                     has_bias=has_bias,
                                     random_state=seed)
    cov = rmatch._covariance()
    for l, v in zip(rmatch.eigvals, rmatch.eigvecs):
        assert np.allclose(cov @ v, l * v)
    for l, v in zip(rmatch.eigvals, rmatch.eigvecs.T):
        assert np.allclose(cov @ v, l * v)


@given(dimensions(), st.booleans(), st.floats(min_value=0.01, max_value=0.99),
       st.floats(min_value=0.01, max_value=0.99), seeds())
def test_volume_after_random_init(dX, has_bias, cover_confidence, coverage,
                                  seed):
    rmatch = RadialMatch.random_ball(dX=dX,
                                     has_bias=has_bias,
                                     cover_confidence=cover_confidence,
                                     coverage=coverage,
                                     random_state=seed)
    vol = rmatch.covered_vol(cover_confidence)
    assert np.isclose(vol, coverage * space_vol(dX))


@given(
    arrays(np.float64, (10, ),
           elements=st.floats(min_value=0.5,
                              max_value=1e10,
                              exclude_min=True,
                              allow_infinity=False,
                              allow_nan=False)),
    st.floats(min_value=0.01, max_value=0.99),
    st.floats(min_value=-0.99, max_value=0.99),
    st.floats(min_value=0.01, max_value=0.99), seeds())
@settings(deadline=None)
def test_volume_after__stretch(eigvals, cover_confidence, pvol, scale, seed):
    vol = radial._covered_vol(eigvals, cover_confidence)
    voldiff = pvol * vol
    eigvals_ = radial._stretch(eigvals,
                               voldiff=voldiff,
                               scale=scale,
                               cover_confidence=cover_confidence,
                               random_state=check_random_state(seed))
    vol_ = radial._covered_vol(eigvals_, cover_confidence)
    voldiff_ = vol_ - vol
    assert np.isclose(voldiff, voldiff_, atol=1e-7, rtol=1e-7)


@given(
    arrays(np.float64, (10, ),
           elements=st.floats(min_value=0.5,
                              max_value=1e10,
                              exclude_min=True,
                              allow_infinity=False,
                              allow_nan=False)),
    st.floats(min_value=0.01, max_value=0.99),
    st.floats(min_value=-0.99, max_value=0.99),
    st.floats(min_value=0.01, max_value=0.99), seeds())
@settings(deadline=None)
def test_isnan_after__stretch(eigvals, cover_confidence, pvol, scale, seed):
    vol = radial._covered_vol(eigvals, cover_confidence)
    voldiff = pvol * vol
    eigvals_ = radial._stretch(eigvals,
                               voldiff=voldiff,
                               scale=scale,
                               cover_confidence=cover_confidence,
                               random_state=check_random_state(seed))
    assert not np.any(np.isnan(eigvals_))


@given(dimensions(), st.floats(min_value=0.01, max_value=0.99),
       st.floats(min_value=0.01, max_value=1.),
       st.floats(min_value=0.01, max_value=0.99), seeds())
def test_volume_after_mutate_large_enough(dX, cover_confidence, coverage, pvol,
                                          seed):
    """
    When the volume is large enough and we can use the normal mutation (without
    clipping or anything).
    """
    rmatch = RadialMatch.random_ball(dX=dX,
                                     has_bias=True,
                                     cover_confidence=cover_confidence,
                                     coverage=coverage,
                                     random_state=seed)
    mutate(rmatch,
           cover_confidence=cover_confidence,
           volstd=pvol,
           random_state=seed + 1)

    vol_ = rmatch.covered_vol(cover_confidence)

    assert vol_ > 0.01


@given(dimensions(), st.floats(min_value=0.01, max_value=0.99))
def chiinv_is_incgammainv(n, confidence):
    # def test_chiinv_is_incgammainv(n, confidence):
    r1 = radius_for_ci(n, confidence)
    r2 = np.sqrt(sst.chi2.ppf(confidence, n))
    assert r1 == r2


def plot_rotation():
    # # This is just for trying it out, should extract.
    # def plot_rotation():
    seed = 1
    dX = 2
    random_state = check_random_state(seed)
    rmatch = RadialMatch.random_ball(dX=dX,
                                     random_state=random_state,
                                     has_bias=False)
    import matplotlib.pyplot as plt
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    z = np.zeros((len(x), len(y)))
    rmatch.eigvals[0] += 1
    print(rmatch.eigvecs)
    for i in range(10):
        # https://stackoverflow.com/questions/1208118/
        for i in range(len(x)):
            for j in range(len(y)):
                X = np.array([[x[i], y[j]]])
                z[i][j] = rmatch.match(X)[0][0]
        # z = rmatch.match(X)
        # breakpoint()
        h = plt.contourf(x, y, z)
        plt.show()
        rmatch = mutate(rmatch, random_state=random_state)
        print(rmatch.eigvecs)
        print(rmatch.eigvecs.T)
        print(np.linalg.inv(rmatch.eigvecs))
