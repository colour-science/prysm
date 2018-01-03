''' This submodule imports and exports math functions from different libraries.
    The intend is to make the backend for prysm interoperable, allowing users to
    utilize more high performance engines if they have them installed, or fall
    back to more widely available options in the case that they do not.
'''

from math import (
    floor,
    ceil,
    pi,
    nan,
)
import numpy as np
from numpy import (
    sqrt,
    sin,
    cos,
    tan,
    arctan,
    arctan2,
    sinc,
    exp,
    log,
    log10,
    arccos,
    arcsin,
)
from numpy.fft import fftshift, ifftshift

atan2 = arctan2
atan = arctan

# numba funcs, cuda
try:
    from numba import jit, vectorize
except ImportError:
    # if Numba is not installed, create the jit decorator and have it return the
    # original function.
    def jit(signature_or_function=None, locals={}, target='cpu', cache=False, **options):
        if signature_or_function is None:
            def _jit(function):
                return function
            return _jit
        else:
            return signature_or_function

    vectorize = jit


# export control
# thanks, ITAR


def fft2(array):
    return np.fft.fft2(array)


def ifft2(array):
    return np.fft.ifft2(array)


# stop pyflakes import errors
assert floor
assert ceil
assert pi
assert nan
assert sin
assert cos
assert tan
assert sinc
assert exp
assert log
assert log10
assert arccos
assert arcsin
assert arctan
assert fftshift
assert ifftshift
assert sqrt
