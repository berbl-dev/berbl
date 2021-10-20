import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given, seed, settings  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.literal.hyperparams import HParams
from prolcs.literal.model import Model
from prolcs.match.allmatch import AllMatch
from prolcs.match.radial1d_drugowitsch import RadialMatch1D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_random_state  # type: ignore

from test_prolcs import rmatch1ds, Xs, ys, seeds

# NOTE Matching functions always assume a bias column (``has_bias=True``)
# whereas the ``Xs`` do not contain one (``bias_column=False``) because
# ``Model.fit`` adds it automatically.


@given(st.lists(rmatch1ds(has_bias=True), min_size=2, max_size=10),
       Xs(bias_column=False), ys())
@settings(deadline=None, max_examples=20)
# hypothesis.errors.FailedHealthCheck: Examples routinely exceeded the max
# allowable size. (20 examples overran while generating 9 valid ones).
# Generating examples this large will usually lead to bad results. You could try
# setting max_size parameters on your collections and turning max_leaves down on
# recursive() calls.
@seed(338435219230913684853574049358930463006)
def test_fit(matchs, X, y):
    """
    Only tests whether fit runs through without errors.
    """
    m = Model(matchs)
    m.fit(X, y)


@given(st.lists(rmatch1ds(has_bias=True), min_size=2, max_size=10),
       Xs(bias_column=False), ys(), Xs(bias_column=False))
@settings(deadline=None, max_examples=20)
# Disable shrink
# phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target))
def test_fit_predict(matchs, X, y, X_test):
    """
    Only tests whether fit and then predict run through without errors.
    """
    m = Model(matchs)
    m.fit(X, y)
    m.predict(X_test)


@st.composite
def linears(draw, N=10, slope_range=(0, 1), intercept_range=(0, 1)):
    """
    Creates a “perfectly” sampled sample for a random affine linear function on
    [-1, 1].
    """
    D_X = 1
    # We create perfect values for X here so we don't run into sampling issues
    # (i.e. evenly spaced).
    X = np.arange(-1, 1, 2 / (N))[:, np.newaxis]

    slope = draw(
        st.floats(min_value=slope_range[0],
                  max_value=slope_range[1],
                  allow_nan=False,
                  allow_infinity=False))
    intercept = draw(
        st.floats(min_value=intercept_range[0],
                  max_value=intercept_range[1],
                  allow_nan=False,
                  allow_infinity=False))

    y = X * slope + intercept

    return (X, y, slope, intercept)


@given(linears(N=10, slope_range=(0, 1), intercept_range=(0, 1)))
@settings(max_examples=50)
def test_fit_linear_functions(data):
    """
    Learning one-dimensional (affine) linear functions should be doable for a
    single classifier that is responsible for/matches all inputs (i.e. it should
    be able to find the slope and intercept).
    """
    X, y, slope, intercept = data

    match = AllMatch()
    # NOTE In `test_fit_non_linear` we use two `AllMatch`s instead of one
    # because otherwise the Laplace approximation may lead to overflows in
    # `train_mix_priors`. Maybe this is required here as well?

    # The default MAX_ITER_CLS seems to be too small for good approximations.
    HParams().MAX_ITER_CLS = 100
    m = Model([match]).fit(X, y)
    HParams().MAX_ITER_CLS = 20  # Reset to the default.

    y_pred = m.predicts(X)[0]

    assert y_pred.shape[1] == 1, (f"Shape of prediction output is wrong "
                                  f"({y.pred.shape[1]} instead of 1)")

    score = mean_absolute_error(y_pred, y)

    # TODO This is not yet ideal; we probably want to scale the score by the
    # range of y values somehow.
    # score /= np.max(y) - np.min(y)
    tol = 1e-2
    assert score < tol, (
        f"Mean absolute error is {score} (> {tol})."
        f"Even though L(q) = {m.L_k_q_}, classifier's weight matrix is still: "
        f"{m.W_[0]} when it should be [{intercept}, {slope}].\n"
        f"Also, predictions are:\n {np.hstack([y, y_pred])}")


@st.composite
def random_data(draw, N=100):
    """
    Creates a “perfectly” sampled sample for a random (non-smooth) function on
    [-1, 1] in 1 to 10 input or output dimensions.
    """
    D_X = draw(st.integers(min_value=1, max_value=10))
    D_Y = draw(st.integers(min_value=1, max_value=10))

    # We create perfect values for X here so we don't run into sampling issues
    # (i.e. evenly spaced).
    X = np.arange(-1, 1, 2 / (N))[:, np.newaxis]

    # Although values for y lie in [-1, 1] we do not standardize them for the
    # sake of this test. (We could, however, by dividing by (1 - (-1))^2 / 12, I
    # think.)
    y = draw(
        arrays(np.float64, (N, D_Y),
               elements=st.floats(min_value=-1, max_value=1)))

    return (X, y)


# We may need to use more samples here to make sure that the algorithms' scores
# are really close.
@given(random_data(N=1000), seeds())
@settings(deadline=None)
def test_fit_non_linear(data, seed):
    """
    A single classifier should behave better or very similar to a
    ``sklearn.linear_model.LinearRegression`` on random data.
    """
    X, y = data

    match = AllMatch()
    # We use two `AllMatch`s because otherwise the Laplace approximation may
    # lead to overflows in `train_mix_priors`.
    # TODO Is this fixable in `train_mix_priors`
    match2 = AllMatch()

    # The default MAX_ITER_CLS seems to be too small for good approximations.
    HParams().MAX_ITER_CLS = 100

    # We seed this so we don't get flaky tests.
    random_state = check_random_state(seed)
    m = Model([match, match2], random_state=random_state).fit(X, y)

    # Reset to the default.
    HParams().MAX_ITER_CLS = 20

    y_pred = m.predicts(X)[0]

    score = mean_absolute_error(y_pred, y)

    oracle = LinearRegression().fit(X, y)
    y_pred_oracle = oracle.predict(X)
    score_oracle = mean_absolute_error(y_pred_oracle, y)

    # We clip scores because of possible floating point instabilities arising in
    # this test if they are too close to zero (i.e. proper comparison becomes
    # inconveniently hard to do).
    score = np.clip(score, a_min=1e-3, a_max=np.inf)
    score_oracle = np.clip(score_oracle, a_min=1e-3, a_max=np.inf)

    # We allow deviations by up to ``(atol + rtol * score_oracle)`` from
    # ``score_oracle``.
    atol = 1e-4
    rtol = 1e-1
    assert (
        score < score_oracle
        or np.isclose(score, score_oracle, atol=atol, rtol=rtol)
    ), (f"Classifier score ({score}) not close to "
        f"linear regression oracle score ({score_oracle}): "
        f"absolute(a - b) <= (atol + rtol * absolute(b)) is "
        f"{np.abs(score - score_oracle)} <= "
        f"({atol} + {rtol * np.abs(score_oracle)}) which is"
        f"{np.abs(score - score_oracle)} <= "
        f"({atol + rtol * np.abs(score_oracle)})"
        )


@given(st.lists(rmatch1ds(has_bias=True), min_size=2, max_size=10),
       Xs(bias_column=False), ys(), seeds())
@settings(deadline=None)
def test_model_fit_deterministic(matchs, X, y, seed):
    random_state = check_random_state(seed)
    m = Model(matchs, random_state=random_state)
    m.fit(X, y)

    random_state2 = check_random_state(seed)
    m2 = Model(matchs, random_state=random_state2)
    m2.fit(X, y)

    for key in m.metrics_:
        assert np.array_equal(m.metrics_[key], m2.metrics_[key])

    for key in m.params_:
        assert np.array_equal(m.params_[key], m2.params_[key])

    # TODO Once in a while (very seldomly, may be fixed already) this throws
    # either of these two
    #
    #   File "prolcs/src/prolcs/literal/__init__.py", line 668, in var_mix_bound
    #     L_M1q = L_M1q + ss.gammaln(a_beta[k]) - a_beta[k] * np.log(b_beta[k])
    # FloatingPointError: invalid value encountered in log
    #
    #   File "prolcs/src/prolcs/literal/__init__.py", line 449, in train_mix_weights
    #     R * np.nan_to_num(np.log(G / R), nan=0, posinf=0, neginf=0))
    # FloatingPointError: overflow encountered in true_divide
