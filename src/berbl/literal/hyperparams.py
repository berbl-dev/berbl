import numpy as np # type: ignore

from ..utils import EXP_MIN, LN_MAX


class HParams():
    """
    We use Alex Martelli's Borg pattern for not having to add Drugowitsch's
    hyper parameters to all signatures.

    See [this](https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/).
    """
    __shared_state = {
        "A_ALPHA": 10**-2,
        "B_ALPHA": 10**-4,
        "A_BETA": 10**-2,
        "B_BETA": 10**-4,
        "A_TAU": 10**-2,
        "B_TAU": 10**-4,
        "DELTA_S_L_K_Q": 10**-4,
        "DELTA_S_L_M_Q": 10**-2,
        "DELTA_S_KLRG": 10**-8,
        "EXP_MIN": EXP_MIN,
        "LN_MAX": LN_MAX,
        # This is not documented in the book but in the code at
        # https://github.com/jdrugo/LCSBookCode
        # TODO Rename to MAX_ITER_RULE
        "MAX_ITER_CLS": 20,
        # This is not documented in the book but in the code at
        # https://github.com/jdrugo/LCSBookCode
        "MAX_ITER_MIXING": 40,
    }

    def __init__(self):
        self.__dict__ = self.__shared_state
