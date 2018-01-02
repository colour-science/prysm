''' Supplimental tools for computing fourier transforms
'''
import numpy as np

from prysm.mathops import (floor, exp, sqrt)


def pad2d(array, factor=1, value=0):
    ''' Symmetrically pads a 2D array with a value.

    Args:
        array (`numpy.ndarray`): source array.

        factor (`number`): number of widths of source array to add to each side (L/R/U/D).

        value (`number`): value with which to pad the array.

    Returns
        `numpy.ndarray`: padded array.

    '''
    x, y = array.shape
    pad_shape = ((int(x * factor), int(x * factor)), (int(y * factor), int(y * factor)))
    return np.pad(array, pad_width=pad_shape, mode='constant', constant_values=value)


def forward_ft_unit(sample_spacing, samples):
    ''' Computes the units resulting from a fourier transform.

    Args:
        sample_spacing (`float`): center-to-center spacing of samples in an array.

        samples (`int`): number of samples in the data.

    Returns:
        `numpy.ndarray`: array of sample frequencies in the output of an fft.

    '''
    return np.fft.ifftshift(np.fft.fftfreq(samples, sample_spacing / 1e3))


def matrix_dft(f, alpha, npix, shift=None, unitary=False):
    '''
    A technique shamelessly stolen from Andy Kee @ NASA JPL
    Is it magic or math?
    '''
    if np.isscalar(alpha):
        ax = ay = alpha
    else:
        ax = ay = np.asarray(alpha)

    f = np.asarray(f)
    m, n = f.shape

    if np.isscalar(npix):
        M = N = npix
    else:
        M = N = np.asarray(npix)

    if shift is None:
        sx = sy = 0
    else:
        sx = sy = np.asarray(shift)

    # Y and X are (r,c) coordinates in the (m x n) input plane, f
    # V and U are (r,c) coordinates in the (M x N) output plane, F
    X = np.arange(n) - floor(n / 2) - sx
    Y = np.arange(m) - floor(m / 2) - sy
    U = np.arange(N) - floor(N / 2) - sx
    V = np.arange(M) - floor(M / 2) - sy

    E1 = exp(1j * -2 * np.pi * (ay / m) * np.outer(Y, V).T)
    E2 = exp(1j * -2 * np.pi * (ax / m) * np.outer(X, U))

    F = E1.dot(f).dot(E2)

    if unitary is True:
        norm_coef = sqrt((ay * ax) / (m * n * M * N))
        return F * norm_coef
    else:
        return F
