import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given, seed, settings  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from prolcs.allmatch import AllMatch
from sklearn.metrics import mean_absolute_error
from prolcs.literal.model import Model
from prolcs.literal.hyperparams import HParams
from prolcs.match.radial1d_drugowitsch import RadialMatch1D
from sklearn.linear_model import LinearRegression


@st.composite
def match1ds(draw):
    a = draw(st.floats(min_value=0, max_value=100))
    b = draw(st.floats(min_value=0, max_value=50))
    return RadialMatch1D(a=a, b=b)


@st.composite
def Xs(draw, N=10, D_X=1):
    X = draw(
        arrays(np.float64, (N, D_X),
               elements=st.floats(min_value=-1, max_value=1),
               unique=True))
    return X


@st.composite
def ys(draw, N=10, D_y=1):
    return draw(
        arrays(np.float64, (N, D_y),
               elements=st.floats(min_value=-1, max_value=1)))


@given(st.lists(match1ds(), min_size=2, max_size=10), Xs(), ys())
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


@given(st.lists(match1ds(), min_size=2, max_size=10), Xs(), ys(), Xs())
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

    y = draw(
        arrays(np.float64, (N, D_Y),
               elements=st.floats(min_value=0, max_value=100)))

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

    # The default MAX_ITER_CLS seems to be too small for good approximations.
    HParams().MAX_ITER_CLS = 100
    m = Model([match]).fit(X, y)
    HParams().MAX_ITER_CLS = 20  # Reset to the default.

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

    assert (score < score_oracle
            or np.isclose(score / score_oracle, 1, atol=1e-1)), (
                f"Classifier score ({score}) not close to"
                f"linear regression oracle score ({score_oracle})")
