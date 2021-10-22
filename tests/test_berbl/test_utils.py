import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis import given  # type: ignore
from berbl.utils import pr_in_sd, radius_for_ci


@given(st.integers(min_value=1, max_value=100),
       st.floats(min_value=1, max_value=100))
def test_pr_in_sd_is_pr(n, r):
    ci = pr_in_sd(n, r)
    assert 0 <= ci <= 1


@given(st.integers(min_value=1, max_value=100),
       st.floats(min_value=0.01, max_value=0.99))
def test_radius_for_ci_finite(n, ci):
    r_ = radius_for_ci(n, ci)
    assert np.isfinite(r_)


@given(
    st.tuples(st.integers(min_value=1, max_value=100),
              st.floats(min_value=1, max_value=10)).
    filter(
        # Only generate data resulting in valid confidence intervals
        # and make sure that we don't run into numerical issues.
        lambda x: 0.01 <= pr_in_sd(x[0], x[1]) <= 0.99))
def test_radius_for_ci_inverse_pr_in_sd(nr):
    n, r = nr

    ci = pr_in_sd(n, r)

    r_ = radius_for_ci(n, ci)

    assert np.isclose(
        r_ / r, 1, rtol=1e-1), (f"not inverse to pr_in_sd: ci = {ci} and thus "
                                f"r' / r = {r_} / {r} = {r_ / r}")


