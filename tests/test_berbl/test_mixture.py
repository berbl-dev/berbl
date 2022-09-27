from copy import copy

import berbl.literal.model as literal
import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from berbl.match.allmatch import AllMatch
from berbl.mixture import Mixture
from berbl.utils import add_bias
from hypothesis import given  # type: ignore; type: ignore
from hypothesis import settings  # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from test_berbl import (Xs, assert_isclose, linears, noshrinking, random_data,
                        random_states, rmatch1ds, ys)


@given(st.lists(rmatch1ds(has_bias=True), min_size=2, max_size=10),
       Xs(bias_column=False), ys(), Xs(bias_column=False), random_states())
@settings(deadline=None)
def test_predict_batch_equals_point(matchs, X, y, X_test, random_state):
    """
    Whether the used batch form of predict equals the point-wise formula from
    the book.
    """
    mixture = Mixture(matchs, random_state=random_state).fit(X, y)

    y_pred, y_pred_var = mixture.predict_mean_var(X_test)

    _, Dy = y.shape
    N = len(X_test)
    rules = mixture.rules_
    K = mixture.K_

    X_test = add_bias(X_test)
    G = mixture.mixing_.mixing(X_test)

    y_pred_ = np.zeros((N, Dy))
    for n in range(N):
        gW = np.zeros(rules[0].W_.shape)
        for k in range(K):
            gW += G[n][k] * rules[k].W_
        y_pred_[n] = gW @ X_test[n]
    assert np.all(np.isclose(y_pred, y_pred_))

    y_pred_var_ = np.zeros((N, Dy))
    for n in range(N):
        x = X_test[n]
        for j in range(Dy):
            for k in range(K):
                cl = rules[k]
                y_pred_var_[n][j] += G[n][k] * (2 * cl.b_tau_ /
                                                (cl.a_tau_ - 1) *
                                                (1 + x.T @ cl.Lambda_1_ @ x) +
                                                (cl.W_[j] @ x)**2)
            y_pred_var_[n][j] -= y_pred_[n][j]**2

    assert np.all(np.isclose(y_pred_var,
                             y_pred_var_)), (y_pred_var - y_pred_var_)


@given(linears(N=100, slope_range=(0, 1), intercept_range=(0, 1)),
       random_states())
@settings(max_examples=50)
def test_fit_linear_functions(data, random_state):
    """
    Learning one-dimensional (affine) linear functions should be doable for a
    single rule that is responsible for/matches all inputs (i.e. it should be
    able to find the slope and intercept).
    """
    X, y, slope, intercept = data

    match = AllMatch()
    mixture = Mixture([match], random_state=random_state).fit(X, y)

    y_pred = mixture.predict(X)

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
@given(random_data(N=1000), random_states())
def test_fit_non_linear(data, random_state):
    """
    A single rule should behave better or very similar to a
    `sklearn.linear_model.LinearRegression` on random data.
    """
    X, y = data

    match = AllMatch()
    mixture = Mixture([match],
                      random_state=random_state,
                      fit_mixing="laplace", MAX_ITER_RULE=200).fit(X, y)

    y_pred = mixture.predict(X)
    score = mean_absolute_error(y_pred, y)

    oracle = LinearRegression().fit(X, y)
    y_pred_oracle = oracle.predict(X)
    score_oracle = mean_absolute_error(y_pred_oracle, y)

    # We clip scores because of possible floating point instabilities arising in
    # this test if they are too close to zero (i.e. proper comparison becomes
    # inconveniently hard to do).
    score = np.clip(score, a_min=1e-4, a_max=np.inf)
    score_oracle = np.clip(score_oracle, a_min=1e-4, a_max=np.inf)

    assert (score < score_oracle
            or np.isclose(score, score_oracle, atol=1e-3)), (
                f"Submodel score ({score}) not close to "
                f"linear regression oracle score ({score_oracle})")


@given(random_data(N=100),
       st.lists(rmatch1ds(has_bias=True), min_size=2, max_size=10),
       random_states())
@settings(deadline=None, phases=noshrinking)
def test_varbounds_like_literal(data, matchs, random_state):

    X, y = data

    mixture = Mixture(matchs,
                      random_state=copy(random_state),
                      add_bias=False,
                      fit_mixing="laplace").fit(X, y)
    model = literal.Model(matchs,
                          random_state=copy(random_state),
                          add_bias=False).fit(X, y)

    assert_isclose(np.sum(mixture.L_C_q_),
                   model.L_C_q_,
                   label="L_C_q",
                   rtol=0.01)
    assert_isclose(mixture.L_M_q_, model.L_M_q_, label="L_M_q_", rtol=0.01)
    assert_isclose(mixture.L_q_, model.L_q_, label="L_q_", rtol=0.01)
    assert_isclose(mixture.p_M_D_, model.p_M_D_, label="p_M_D_", rtol=0.01)
