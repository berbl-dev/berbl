import berbl.literal as literal
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from berbl.match.allmatch import AllMatch
from berbl.rule import Rule
from hypothesis import given, settings  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from test_berbl import (Xs, assert_isclose, linears, noshrinking, random_data,
                        ys)

# TODO Expand to n-dim radial matching. Currently this is only for
# one-dimensional data (possibly with a bias column)


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
        L_qs[i] = Rule(match, MAX_ITER=max_iters[i]).fit(X, y).L_q_

    assert np.all(np.diff(L_qs) >= 0)


@given(linears(N=10, slope_range=(0, 1), intercept_range=(0, 1)))
@settings(max_examples=50)
def test_fit_linear_functions(data):
    """
    Learning one-dimensional (affine) linear functions should be doable for a
    single rule that is responsible for/matches all inputs (i.e. it should be
    able to find the slope and intercept).
    """
    X, y, slope, intercept = data

    match = AllMatch()

    cl = Rule(match, MAX_ITER=100).fit(X, y)

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
        f"Even though L(q) = {cl.L_q_}, rule's weight matrix is still"
        f"{cl.W_} when it should be [{intercept}, {slope}]")


# We use more samples here to make sure that the algorithms' score are
# really close.
@given(random_data(N=1000))
def test_fit_non_linear(data):
    """
    A single rule should behave better or very similar to a
    ``sklearn.linear_model.LinearRegression`` on random data.
    """
    X, y = data

    match = AllMatch()

    cl = Rule(match, MAX_ITER_RULE=200).fit(X, y)

    y_pred = cl.predict(X)
    score = mean_absolute_error(y_pred, y)

    oracle = LinearRegression().fit(X, y)
    y_pred_oracle = oracle.predict(X)
    score_oracle = mean_absolute_error(y_pred_oracle, y)

    # We clip scores because of possible floating point instabilities arising in
    # this test if they are too close to zero (i.e. proper comparison becomes
    # inconveniently hard to do).
    score = np.clip(score, a_min=1e-4, a_max=np.inf)
    score_oracle = np.clip(score_oracle, a_min=1e-4, a_max=np.inf)

    if score > score_oracle:
        assert_isclose(score, score_oracle, atol=1e-3)


@given(random_data())
def test_predict_var_batch_equals_point(data):
    """
    Vectorized form of predict_var equals point-wise form.
    """
    X, y = data
    match = AllMatch()
    cl = Rule(match, MAX_ITER=100).fit(X, y)

    Dy = cl.Dy_
    y_var = cl.predict_var(X)
    assert y_var.shape == (len(X), Dy)

    for n in range(len(X)):
        for j in range(Dy):
            y_var_n_j = 2 * cl.b_tau_ / (cl.a_tau_ - 1) * (
                1 + X[n].T @ cl.Lambda_1_ @ X[n])
            assert np.isclose(y_var[n][j], y_var_n_j, atol=1e-3,
                              rtol=1e-3), (y_var[n][j] - y_var_n_j)


# DX = 5, N = 10
@given(Xs(N=10, DX=5, bias_column=False),
       arrays(np.float64, (5, 5),
              elements=st.floats(min_value=0,
                                 max_value=100,
                                 allow_infinity=False,
                                 allow_nan=False)))
def test_square_with_matrix(X, Lambda_1_):
    y1 = np.sum((X @ Lambda_1_) * X, axis=1)
    y2 = np.zeros((10, ))
    y3 = np.diag(X @ Lambda_1_ @ X.T)
    for n in range(10):
        y2[n] = X[n].T @ Lambda_1_ @ X[n]

    assert np.all(np.isclose(y1, y2, rtol=1e-7)), (y1 - y2)
    assert np.all(np.isclose(y1, y3, rtol=1e-7)), (y1 - y3)


@given(random_data(N=100))
@settings(deadline=None, phases=noshrinking)
def test_fit_like_literal(data):
    X, y = data
    match = AllMatch()

    rule = Rule(match).fit(X, y)

    m_k = match.match(X)
    W_k, Lambda_1_k, a_tau_k, b_tau_k, a_alpha_k, b_alpha_k = literal.train_classifier(
        m_k, X, y)

    assert_isclose(rule.W_, W_k, label="W_")
    assert_isclose(rule.Lambda_1_, Lambda_1_k, label="Lambda_1_")
    assert_isclose(rule.a_tau_, a_tau_k, label="a_tau_")
    assert_isclose(rule.b_tau_, b_tau_k, label="b_tau_")
    assert_isclose(rule.a_alpha_, a_alpha_k, label="a_alpha_")
    assert_isclose(rule.b_alpha_, b_alpha_k, label="b_alpha_")


# TODO Add tests for all the other hyperparameters of Rule.
