import numpy as np  # type: ignore
from hypothesis import given  # type: ignore
from test_berbl import Xs_and_match1ds, imatch1ds


@given(Xs_and_match1ds(imatch1ds))
def test_match_never_nan(X_and_match1d):
    X, match = X_and_match1d
    assert np.all(~np.isnan(match.match(X)))


@given(Xs_and_match1ds(imatch1ds))
def test_match_prob_bounds(X_and_match1d):
    X, match = X_and_match1d
    m = match.match(X)
    # All matching functions should match all samples, at least a little bit.
    assert np.all(0 < m)
    assert np.all(m <= 1)
