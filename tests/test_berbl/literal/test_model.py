import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from berbl.literal import mixing
from berbl.literal.hyperparams import HParams
from berbl.literal.model import Model
from berbl.match.allmatch import AllMatch
from berbl.utils import add_bias, check_phi, matching_matrix
from hypothesis import given, settings  # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_random_state  # type: ignore
from test_berbl import (Xs, assert_isclose, linears, random_data,
                        random_states, rmatch1ds, seeds, ys)

# NOTE Matching functions always assume a bias column (``has_bias=True``)
# whereas the ``Xs`` do not contain one (``bias_column=False``) because
# ``Model.fit`` adds it automatically.


@given(linears(N=10, slope_range=(0, 1), intercept_range=(0, 1)), random_states())
@settings(max_examples=50)
def test_fit_linear_functions(data, random_state):
    """
    Learning one-dimensional (affine) linear functions should be doable for a
    single rule that is responsible for/matches all inputs (i.e. it should be
    able to find the slope and intercept).
    """
    X, y, slope, intercept = data

    match = AllMatch()
    # NOTE In `test_fit_non_linear` we use two `AllMatch`s instead of one
    # because otherwise the Laplace approximation may lead to overflows in
    # `train_mix_priors`. Maybe this is required here as well?

    # The default MAX_ITER_CLS seems to be too small for good approximations.
    HParams().MAX_ITER_CLS = 100
    m = Model([match], random_state=random_state).fit(X, y)
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
        f"Even though L(q) = {m.L_C_q_}, submodel's weight matrix is still: "
        f"{m.W_[0]} when it should be [{intercept}, {slope}].\n"
        f"Also, predictions are:\n {np.hstack([y, y_pred])}")


# We may need to use more samples here to make sure that the algorithms' scores
# are really close.
@given(random_data(N=1000, bias_column=False), random_states())
# Increase number of tests in order to catch numerical issues that happen
# seldomly.
@settings(deadline=None, max_examples=500)
def test_fit_non_linear(data, random_state):
    """
    A single rule should behave better or very similar to a
    ``sklearn.linear_model.LinearRegression`` on random data.
    """
    X, y = data

    match = AllMatch()

    # The default MAX_ITER_CLS seems to be too small for good approximations.
    HParams().MAX_ITER_CLS = 200

    m = Model([match], random_state=random_state).fit(X, y)

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
    atol = 1e-3
    if score > score_oracle:
        assert_isclose(score, score_oracle, atol=atol)


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

@given(st.lists(rmatch1ds(has_bias=True), min_size=2, max_size=10),
       Xs(bias_column=False), ys(), Xs(bias_column=False), random_states())
@settings(deadline=None)
def test_predict_batch_equals_point(matchs, X, y, X_test, random_state):
    """
    Whether the used batch form of predict equals the point-wise formula from
    the book.
    """
    model = Model(matchs, random_state=random_state).fit(X, y)

    y_pred, y_pred_var = model.predict_mean_var(X_test)

    _, Dy = y.shape
    N = len(X_test)
    K = model.K_

    X_test = add_bias(X_test)
    M = matching_matrix(matchs, X_test)
    Phi = check_phi(model.phi, X_test)
    G = mixing(M=M, Phi=Phi, V=model.V_)

    y_pred_ = np.zeros((N, Dy))
    for n in range(N):
        gW = np.zeros(model.W_[0].shape)
        for k in range(K):
            gW += G[n][k] * model.W_[k]
        y_pred_[n] = gW @ X_test[n]
    assert np.all(np.isclose(y_pred, y_pred_))

    y_pred_var_ = np.zeros((N, Dy))
    for n in range(N):
        x = X_test[n]
        for j in range(Dy):
            for k in range(K):
                y_pred_var_[n][j] += G[n][k] * (2 * model.b_tau_[k] /
                                                (model.a_tau_[k] - 1) *
                                                (1 + x.T @ model.Lambda_1_[k] @ x) +
                                                (model.W_[k][j] @ x)**2)
            y_pred_var_[n][j] -= y_pred_[n][j]**2

    assert np.all(np.isclose(y_pred_var,
                             y_pred_var_)), (y_pred_var - y_pred_var_)
