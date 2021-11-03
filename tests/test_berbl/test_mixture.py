import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from berbl.mixture import Mixture
from berbl.utils import add_bias
from hypothesis import given  # type: ignore
from hypothesis import settings  # type: ignore
from test_berbl import Xs, random_states, rmatch1ds, ys


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
