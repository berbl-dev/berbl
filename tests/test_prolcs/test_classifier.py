# import pytest  # type: ignore
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given, settings  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.match.allmatch import AllMatch
from prolcs.classifier import Classifier
from prolcs.match.radial1d_drugowitsch import RadialMatch1D
from prolcs.utils import add_bias
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


# TODO Expand to n-dim radial matching. Currently this is only for
# one-dimensional data (possibly with a bias column)


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


@given(Xs(), ys())
@settings(deadline=None, max_examples=15)
def test_fit_inc_L_q(X, y):
    """
    “Each parameter update either increases L_q or leaves it unchanged (…). If
    this is not the case, then the implementation is faulty and/or suffers from
    numerical instabilities.” [PDF p. 237]

    assert delta_L_q >= 0, f"delta_L_q = {delta_L_q} < 0"
    """
    match = AllMatch()
    max_iters = range(1, 101, 10)
    L_qs = np.array([None] * len(max_iters))
    for i in range(len(max_iters)):
        L_qs[i] = Classifier(match, MAX_ITER=max_iters[i]).fit(X, y).L_q_

    assert np.all(np.diff(L_qs) >= 0)


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
    X = add_bias(X)

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

    cl = Classifier(match, MAX_ITER=100).fit(X, y)

    y_pred = cl.predict(X)

    assert y_pred.shape[1] == 1, (f"Shape of prediction output is wrong "
                                  f"({y.pred.shape[1]} instead of 1)")

    score = mean_absolute_error(y_pred, y)

    # TODO This is not yet ideal; we probably want to scale the score by the
    # range of y values somehow.
    # score /= np.max(y) - np.min(y)
    tol = 1e-2
    assert score < tol, (
        f"Mean absolute error is {score} (> {tol})."
        f"Even though L(q) = {cl.L_q_}, classifier's weight matrix is still"
        f"{cl.W_} when it should be [{intercept}, {slope}]")


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

    y = draw(
        arrays(np.float64, (N, D_Y),
               elements=st.floats(min_value=0, max_value=100)))
    X = add_bias(X)

    return (X, y)


# We use more samples here to make sure that the algorithms' score are
# really close.
@given(random_data(N=1000))
def test_fit_non_linear(data):
    """
    A single classifier should behave better or very similar to a
    ``sklearn.linear_model.LinearRegression`` on random data.
    """
    X, y = data

    match = AllMatch()

    cl = Classifier(match, MAX_ITER=100).fit(X, y)

    y_pred = cl.predict(X)
    score = mean_absolute_error(y_pred, y)

    oracle = LinearRegression().fit(X, y)
    y_pred_oracle = oracle.predict(X)
    score_oracle = mean_absolute_error(y_pred_oracle, y)

    # We clip scores because of possible floating point instabilities arising in
    # this test if they are too close to zero (i.e. proper comparison becomes
    # inconveniently hard to do).
    score = np.clip(score, a_min=1e-3, a_max=np.inf)
    score_oracle = np.clip(score_oracle, a_min=1e-3, a_max=np.inf)

    assert (score < score_oracle
            or np.isclose(score / score_oracle, 1, atol=1e-1)), (
                f"Classifier score ({score}) not close to"
                f"linear regression oracle score ({score_oracle})")


# TODO Add tests for all the other hyperparameters of Classifier.
