import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from berbl.match.hardinterval import HardInterval
from berbl.match.radial1d_drugowitsch import RadialMatch1D
from berbl.match.softinterval1d_drugowitsch import SoftInterval1D
from berbl.utils import add_bias
from hypothesis import Phase  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

# TODO Ensure that test data is standardized


@st.composite
def seeds(draw):
    # Highest possible seed is `2**32 - 1` for NumPy legacy generators.
    return draw(st.integers(min_value=0, max_value=2**32 - 1))


@st.composite
def random_states(draw):
    seed = draw(seeds())
    return check_random_state(seed)


@st.composite
def rmatch1ds(draw, has_bias=True):
    a = draw(st.floats(min_value=0, max_value=100))
    b = draw(st.floats(min_value=0, max_value=50))
    return RadialMatch1D(a=a, b=b, has_bias=has_bias)


@st.composite
def imatch1ds(draw, has_bias=True):
    l_ = draw(st.floats(min_value=-1, max_value=1))
    u_ = draw(st.floats(min_value=-1, max_value=1).filter(lambda u_: u_ != l_))
    l = min(l_, u_)
    u = max(l_, u_)
    return SoftInterval1D(l=l, u=u, has_bias=has_bias)


@st.composite
def himatchs(draw, has_bias=True):
    # TODO Add other constructor parameters here
    DX = draw(st.integers(min_value=1, max_value=10))
    random_state = draw(random_states())
    return HardInterval.random(DX=DX,
                               has_bias=has_bias,
                               random_state=random_state)


@st.composite
def Xs(draw, N=10, DX=1, bias_column=True):
    X = draw(
        arrays(np.float64, (N, DX),
               elements=st.floats(min_value=-1, max_value=1),
               unique=True))
    if bias_column:
        X = add_bias(X)
    return X


@st.composite
def ys(draw, N=10, Dy=1):
    return draw(
        arrays(np.float64, (N, Dy),
               elements=st.floats(min_value=-1, max_value=1)))


@st.composite
def Xs_and_match1ds(draw, matchgen, N=10, DX=1):
    """
    Generator for input matrices and match functions that respect whether the
    input matrix contains a bias column or not.

    Parameters
    ----------
    matchgen
        Match function test case generator (probably `rmatch1ds`, `imatch1ds` or
        `himatchs`).
    """
    bias_column = draw(st.booleans())
    X = draw(Xs(N=N, DX=DX, bias_column=bias_column))
    rmatch1d = draw(matchgen(has_bias=bias_column))
    return X, rmatch1d


@st.composite
def Xs_and_matchs(draw, matchgen, N=10):
    """
    Generator for input matrices and match functions that respect whether the
    input matrix contains a bias column or not. The input dimension is drawn at
    random using a Uniform(1, 10) distribution.

    Parameters
    ----------
    matchgen
        Match function test case generator (probably `rmatch1ds`, `imatch1ds` or
        `himatchs`).
    """
    DX = draw(st.integers(min_value=1, max_value=10))
    bias_column = draw(st.booleans())
    X = draw(Xs(N=N, DX=DX, bias_column=bias_column))
    random_state = draw(random_states())
    match = HardInterval.random(DX=DX,
                                has_bias=bias_column,
                                random_state=random_state)
    return X, match


@st.composite
def linears(draw, N=10, slope_range=(0, 1), intercept_range=(0, 1)):
    """
    Creates a “perfectly” sampled sample for a random affine linear function on
    [-1, 1].
    """
    DX = 1
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


@st.composite
def random_data(draw, N=100, bias_column=True):
    """
    Creates a “perfectly” sampled sample for a random (non-smooth) function on
    [-1, 1] in 1 to 10 input or output dimensions.
    """
    DX = draw(st.integers(min_value=1, max_value=10))
    Dy = draw(st.integers(min_value=1, max_value=10))

    # We create perfect values for X here so we don't run into sampling issues
    # (i.e. evenly spaced).
    X = np.arange(-1, 1, 2 / (N))[:, np.newaxis]

    y = draw(
        arrays(np.float64, (N, Dy),
               elements=st.floats(min_value=-1, max_value=1)))
    if bias_column:
        X = add_bias(X)

    return (X, y)


def assert_isclose(a, b, label="", rtol=1e-5, atol=1e-8):
    s = (f"{label} {a} not close enough to {b} "
         f"(after subtracting atol={atol}, "
         f"distance is still {np.abs(a-b) - atol} "
         f"which corresponds to {(np.abs(a-b) - atol) / np.abs(b)} "
         f" >= {rtol}=rtol)")
    assert np.all(np.isclose(a, b, rtol=rtol, atol=atol)), s


"""
The default phases but without shrinking.
"""
noshrinking = ((Phase.explicit, Phase.reuse, Phase.generate, Phase.target))
