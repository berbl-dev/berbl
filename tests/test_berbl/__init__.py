import hypothesis.strategies as st  # type: ignore
import numpy as np  # type: ignore
from hypothesis.extra.numpy import arrays  # type: ignore
from berbl.match.radial1d_drugowitsch import RadialMatch1D
from berbl.match.softinterval1d_drugowitsch import SoftInterval1D
from berbl.utils import add_bias
from sklearn.utils import check_random_state  # type: ignore


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
        Match function test case generator (probably ``rmatch1ds`` or ``imatch1ds``).
    """
    bias_column = draw(st.booleans())
    X = draw(Xs(N=N, DX=DX, bias_column=bias_column))
    rmatch1d = draw(matchgen(has_bias=bias_column))
    return X, rmatch1d
