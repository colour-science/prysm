''' optional tools for colorimetry, wrapping the color-science library, see:
    http://colour-science.org/
'''
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from scipy.constants import c, h, k
# c - speed of light
# h - planck constant
# k - boltzman constant

from matplotlib.collections import LineCollection

from prysm.conf import config
from prysm.util import share_fig_ax, colorline, smooth
from prysm.mathops import atan2, pi, cos, sin, exp, sqrt, arccos, jit

# some CIE constants
CIE_K = 24389 / 27
CIE_E = 216 / 24389

# from Ohno PDF, see D_uv function.
CIE_DUV_k0 = -0.471106
CIE_DUV_k1 = +1.925865
CIE_DUV_k2 = -2.4243787
CIE_DUV_k3 = +1.5317403
CIE_DUV_k4 = -0.5179722
CIE_DUV_k5 = +0.0893944
CIE_DUV_k6 = -0.00616793


# sRGB conversion matrix
XYZ_to_sRGB_mat_D65 = np.asarray([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
])
XYZ_to_sRGB_mat_D50 = np.asarray([
    [3.1338561, -1.6168667, -0.4906146],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0719453, -0.2289914, 1.4052427],
])

# Adobe RGB 1998 matricies
XYZ_to_AdobeRGB_mat_D65 = np.asarray([
    [2.0413690, -0.5649464, -0.3446944],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0134474, -0.1183897, 1.0154096],
])
XYZ_to_AdobeRGB_mat_D50 = np.asarray([
    [1.9624274, -0.6105343, -0.3413404],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0286869, -0.1406752, 1.3487655],
])

COLOR_MATRICIES = {
    'sRGB': {
        'D65': XYZ_to_sRGB_mat_D65,
        'D50': XYZ_to_sRGB_mat_D50,
    },
    'AdobeRGB': {
        'D65': XYZ_to_AdobeRGB_mat_D65,
        'D50': XYZ_to_AdobeRGB_mat_D50,
    },
}

# standard illuminant information
CIE_ILLUMINANT_METADATA = {
    'files': {
        'A': 'cie_A_300_830_1nm.csv',
        'B': 'cie_B_380_770_5nm.csv',
        'C': 'cie_C_380_780_5nm.csv',
        'D': 'cie_Dseries_380_780_5nm.csv',
        'E': 'cie_E_380_780_5nm.csv',
        'F': 'cie_Fseries_380_730_5nm.csv',
        'HP': 'cie_HPseries_380_780_5nm.csv',
    },
    'columns': {
        'A': 1,
        'B': 1,
        'C': 1,
        'D50': 1,
        'D55': 2,
        'D65': 3,
        'E': 1,
        'F1': 1,
        'F2': 2,
        'F3': 3,
        'F4': 4,
        'F5': 5,
        'F6': 6,
        'F7': 7,
        'F8': 8,
        'F9': 9,
        'F10': 10,
        'F11': 11,
        'F12': 12,
        'HP1': 1,
        'HP2': 2,
        'HP3': 3,
        'HP4': 4,
        'HP5': 5,
    }
}


@lru_cache()
def prepare_robertson_cct_data():
    ''' Prepares Robertson's correlated color temperature data.

    Returns:
        `dict` containing: urd, K, u, v, dvdu.

    Notes:
        # CCT values in L*u*v* coordinates, see the following for the source of these values:
        # https://www.osapublishing.org/josa/abstract.cfm?uri=josa-58-11-1528
    '''
    tmp_list = []
    p = Path(__file__).parent / 'color_data' / 'robertson_cct.csv'
    with open(p, 'r') as fid:
        reader = csv.reader(fid)
        for row in reader:
            tmp_list.append(row)

    values = np.asarray(tmp_list[1:], dtype=config.precision)
    urd, k, u, v, dvdu = values[:, 0], values[:, 1], values[:, 2], values[:, 3], values[:, 4]
    return {
        'urd': urd,
        'K': k,
        'u': u,
        'v': v,
        'dvdu': dvdu
    }


def prepare_illuminant_spectrum(illuminant='D65', bb_wvl=None, bb_norm=True):
    ''' Prepares the SPD for a given illuminant.

    Args:
        illuminant (`str`): one of (A, B, C, D50, D55, D65, E, F1..F12, HP1..HP5)
                            or bb_xxxx, for "blackbody_temperature."

        bb_wvl (`numpy.ndarray`): array of wavelengths to compute a requested black body
                                  spd at.

        bb_norm (`bool`): whether to normalize a computed blackbody spectrum.

    Returns:
        `dict` with keys: `wvl`, `values`

    '''
    if illuminant[0:2].lower() == 'bb':
        _, temp = illuminant.split('_')
        if bb_wvl is None:
            bb_wvl = np.arange(380, 780, 5, dtype=config.precision)
        spd = blackbody_spectrum(float(temp), bb_wvl)
        spec = {
            'wvl': bb_wvl,
            'values': spd
        }
        if bb_norm is True:
            spec = normalize_spectrum(spec, to='peak 560')
            spec['values'] *= 100
            return spec
        else:
            return spec
    else:
        return _prepare_ciesource_spectrum(illuminant)


@lru_cache()
def _prepare_ciesource_spectrum(illuminant):
    ''' Retrives a CIE standard source from its csv file.

    Args:
        illuminant (`str`): one of (A, B, C, D50, D55, D65, E, F1..F12, HP1..HP5)
                                or bb_xxxx, for "blackbody_temperature."

    Returns:
        `dict` with keys: `wvl`, `values`

    '''
    if illuminant[0:2].upper() == 'HP':
        file = CIE_ILLUMINANT_METADATA['files']['HP']
    else:
        file = CIE_ILLUMINANT_METADATA['files'][illuminant[0].upper()]
    column = CIE_ILLUMINANT_METADATA['columns'][illuminant.upper()]

    tmp_list = []
    p = Path(__file__).parent / 'color_data' / file
    with open(p, 'r') as fid:
        reader = csv.reader(fid)
        for row in reader:
            tmp_list.append(row)

    values = np.asarray(tmp_list[1:], dtype=config.precision)
    return {
        'wvl': values[:, 0],
        'values': values[:, column],
    }


def value_array_to_tristimulus(values):
    ''' Pulls tristimulus data as numpy arrays from a list of CSV rows.

    Args:
        values (`list`): list with each element being a row of a CSV, headers omitted.

    Returns:
        `dict` with keys: wvl, X, Y, Z

    '''
    values = np.asarray(values, dtype=config.precision)
    wvl, X, Y, Z = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    return {
        'wvl': wvl,
        'X': X,
        'Y': Y,
        'Z': Z
    }


# these two functions could be better refactored, but meh.
@lru_cache()
def prepare_cie_1931_2deg_observer():
    ''' Prepares the CIE 1931 standard 2 degree observer.

    Returns:
        `dict` with keys: wvl, X, Y, Z.

    '''
    tmp_list = []
    p = Path(__file__).parent / 'color_data' / 'cie_xyz_1931_2deg_tristimulus_5nm.csv'
    with open(p, 'r') as fid:
        reader = csv.reader(fid)
        for row in reader:
            tmp_list.append(row)

    return value_array_to_tristimulus(tmp_list[1:])


@lru_cache()
def prepare_cie_1964_10deg_observer():
    ''' Prepares the CIE 1964 standard 10 degree observer.

    Returns:
        `dict` with keys: wvl, X, Y, Z.

    '''
    tmp_list = []
    p = Path(__file__).parent / 'color_data' / 'cie_xyz_1964_10deg_tristimulus_5nm.csv'
    with open(p, 'r') as fid:
        reader = csv.reader(fid)
        for row in reader:
            tmp_list.append(row)

    return value_array_to_tristimulus(tmp_list[1:])


def prepare_cmf(observer='1931_2deg'):
    ''' Safely returns the color matching function dictionary for the specified
        observer.

    Args:
        observer (`str`): the observer to return - either `1931_2deg` or `1964_10deg`.

    Returns:
        `dict`: cmf dict.

    '''
    if observer.lower() == '1931_2deg':
        return prepare_cie_1931_2deg_observer()
    elif observer.lower() == '1964_10deg':
        return prepare_cie_1964_10deg_observer()
    else:
        raise ValueError('observer must be 1931_2deg or 1964_10deg')


def plot_spectrum(spectrum_dict, xrange=(370, 730), yrange=(0, 100), smoothing=None, fig=None, ax=None):
    ''' Plots the spectrum.

    Args:
        xrange (`iterable`): pair of lower and upper x bounds.

        yrange (`iterable`): pair of lower and upper y bounds.

        smoothing (`float`): number of nanometers to smooth data by.  If None,
            do no smoothing.

        fig (`matplotlib.figure.Figure`): figure to plot in.

        ax (`matplotlib.axes.Axis`): axis to plot in.

    Returns:
        `tuple` containing:

            `matplotlib.figure.Figure`: figure containign the plot.

            `matplotlib.axes.Axis`: axis containing the plot.

    '''
    wvl, values = spectrum_dict['wvl'], spectrum_dict['values']
    if smoothing is not None:
        dx = wvl[1] - wvl[0]
        window_width = int(smoothing / dx)
        values = smooth(values, window_width, window='flat')

    lc = colorline(wvl, values, wvl, cmin=400, cmax=700, lw=5)
    fig, ax = share_fig_ax(fig, ax)
    ax.add_collection(lc)
    ax.set(xlim=xrange, xlabel=r'Wavelength $\lambda$ [nm]',
           ylim=yrange, ylabel='Transmission [%]')

    return fig, ax


def blackbody_spectrum(temperature, wavelengths):
    ''' Computes the spectral power distribution of a black body at a given
        temperature.

    Args:
        temperature (`float`): body temp, in Kelvin.

        wavelengths (`numpy.ndarray`): array of wavelengths, in nanometers.

    Returns:
        numpy.ndarray: spectral power distribution in units of W/m^2/nm

    '''
    wavelengths = wavelengths.astype(config.precision) / 1e9
    return (2 * h * c ** 2) / (wavelengths ** 5) * \
        1 / (exp((h * c) / (wavelengths * k * temperature) - 1))


def normalize_spectrum(spectrum, to='peak vis'):
    ''' Normalizes a spectrum to have unit peak within the visible band.
    Args:
        spectrum (`Spectrum`): object with iterable wavelength, value fields.

        to (`str`): either `peak vis` or `peak 560`.
                    'peak vis' - peak falls within visible range.
                    'peak 560' - peak at 560nm.

    Returns:
        `Spectrum`: new spectrum object.

    '''
    wvl, vals = spectrum['wvl'], spectrum['values']
    if to.lower() == 'peak vis':
        low, high = np.searchsorted(wvl, 400), np.searchsorted(wvl, 700)
        vals2 = vals / vals[low:high].max()
    elif to.lower() in ('peak 560', '560', '560nm'):
        idx = np.searchsorted(wvl, 560)
        vals2 = vals / vals[idx]
    return {
        'wvl': wvl,
        'values': vals2,
    }


@lru_cache()
def render_cie_1931_background(xlow, xhigh, ylow, yhigh, samples):
    ''' Prepares the background for a CIE 1931 plot.

    Args:
        xlow (`iterable`): left bound of the image.

        xhigh (`iterable`): right bound of the image.

        ylow (`iterable`): lower bound of the image.

        yhigh (`iterable`): upper bound of the image.

        samples (`int`): number of 1D samples within the region of interest,
            total pixels will be samples^2.

    Returns:
        `numpy.ndarray`: 3D array of sRGB values in the range [0,1], [:,:,[R,G,B]].

    '''
    wvl_mask = [400, 430, 460, 465, 470, 475, 480, 485, 490, 495,
                500, 505, 510, 515, 520, 525, 530, 540, 555, 570, 700]

    wvl_mask_xy = XYZ_to_xy(wavelength_to_XYZ(wvl_mask))

    # make equally spaced u,v coordinates on a grid
    x = np.linspace(xlow, xhigh, samples)
    y = np.linspace(ylow, yhigh, samples)
    xx, yy = np.meshgrid(x, y)

    # stack u and v for vectorized computations, also mask out negative values
    xxyy = np.stack((xx, yy), axis=2)

    # make a mask, of value 1 outside the horseshoe, 0 inside
    triangles = Delaunay(wvl_mask_xy, qhull_options='QJ Qf')
    wvl_mask = triangles.find_simplex(xxyy) < 0

    xyz = xy_to_XYZ(xxyy)
    data = XYZ_to_sRGB(xyz)

    # normalize and clip sRGB values.
    maximum = np.max(data, axis=-1)
    maximum[maximum == 0] = 1
    data = np.clip(data / maximum[:, :, np.newaxis], 0, 1)

    # now make an alpha/transparency mask to hide the background
    alpha = np.ones((samples, samples))
    alpha[wvl_mask] = 0
    data = np.dstack((data, alpha))
    return data


def cie_1931_plot(xlim=(0, 0.9), ylim=None, samples=300, fig=None, ax=None):
    ''' Creates a CIE 1931 plot.

    Args:

        xlim (`iterable`): left and right bounds of the plot.

        ylim (`iterable`): lower and upper bounds of the plot.  If `None`,
            the y bounds will be chosen to match the x bounds.

        samples (`int`): number of 1D samples within the region of interest,
            total pixels will be samples^2.

        fig (`matplotlib.figure.Figure`): figure to plot in.

        ax (`matplotlib.axes.Axis`): axis to plot in.

    Returns:
        `tuple` containing:

            `matplotlib.figure.Figure`: figure containing the plot.

            `matplotlib.axes.axis`: axis containing the plot.

    '''
    # duplicate xlim if ylim not set
    if ylim is None:
        ylim = xlim

    # don't compute over dead space
    xlim_bg = list(xlim)
    ylim_bg = list(ylim)
    if xlim[0] < 0:
        xlim_bg[0] = 0
    if xlim[1] > 0.75:
        xlim_bg[1] = 0.75
    if ylim[0] < 0:
        ylim_bg[0] = 0
    if ylim[1] > 0.85:
        ylim_bg[1] = 0.85

    # create lists of wavelengths and map them to uv,
    # a reduced set for a faster mask and
    # yet another set for annotation.
    wvl_line = np.arange(400, 700, 2.5)
    wvl_line_xy = XYZ_to_xy(wavelength_to_XYZ(wvl_line))

    wvl_annotate = [360, 400, 455, 470, 480, 490,
                    500, 510, 520, 540, 555, 570, 580, 590,
                    600, 615, 630, 700, 830]

    data = render_cie_1931_background(*xlim_bg, *ylim_bg, samples)

    # duplicate the lowest wavelength so that the boundary line is closed
    wvl_line_xy = np.vstack((wvl_line_xy, wvl_line_xy[0, :]))

    fig, ax = share_fig_ax(fig, ax)
    ax.imshow(data,
              extent=[*xlim_bg, *ylim_bg],
              interpolation='bilinear',
              origin='lower')
    ax.plot(wvl_line_xy[:, 0], wvl_line_xy[:, 1], ls='-', c='0.25', lw=2)
    fig, ax = cie_1931_wavelength_annotations(wvl_annotate, fig=fig, ax=ax)
    ax.set(xlim=xlim, xlabel='CIE x',
           ylim=ylim, ylabel='CIE y')

    return fig, ax


@lru_cache()
def render_cie_1976_background(xlow, xhigh, ylow, yhigh, samples):
    ''' Prepares the background for a CIE 1976 plot.

    Args:
        xlow (`iterable`): left bound of the image.

        xhigh (`iterable`): right bound of the image.

        ylow (`iterable`): lower bound of the image.

        yhigh (`iterable`): upper bound of the image.

        samples (`int`): number of 1D samples within the region of interest,
            total pixels will be samples^2.

    Returns:
        `numpy.ndarray`: 3D array of sRGB values in the range [0,1]
                         with shape [samples,samples,[R,G,B]].

    '''
    wvl_mask = [400, 430, 460, 465, 470, 475, 480, 485, 490, 495,
                500, 505, 510, 515, 520, 525, 530, 535, 570, 700]

    wvl_mask_uv = XYZ_to_uvprime(wavelength_to_XYZ(wvl_mask))

    # make equally spaced u,v coordinates on a grid
    u = np.linspace(xlow, xhigh, samples)
    v = np.linspace(ylow, yhigh, samples)
    uu, vv = np.meshgrid(u, v)

    # stack u and v for vectorized computations, also mask out negative values
    uuvv = np.stack((uu, vv), axis=2)

    # make a mask, of value 1 outside the horseshoe, 0 inside
    triangles = Delaunay(wvl_mask_uv, qhull_options='QJ Qf')
    wvl_mask = triangles.find_simplex(uuvv) < 0

    xy = uvprime_to_xy(uuvv)
    xyz = xy_to_XYZ(xy)
    data = XYZ_to_sRGB(xyz)

    # normalize and clip sRGB values.
    maximum = np.max(data, axis=-1)
    maximum[maximum == 0] = 1
    data = np.clip(data / maximum[:, :, np.newaxis], 0, 1)

    # now make an alpha/transparency mask to hide the background
    alpha = np.ones((samples, samples))
    alpha[wvl_mask] = 0
    data = np.dstack((data, alpha))
    return data


def cie_1976_plot(xlim=(-0.09, 0.68), ylim=None, samples=400,
                  annotate_wvl=True, draw_plankian_locust=False,
                  fig=None, ax=None):
    ''' Creates a CIE 1976 plot.

    Args:

        xlim (`iterable`): left and right bounds of the plot.

        ylim (`iterable`): lower and upper bounds of the plot.  If `None`,
            the y bounds will be chosen to match the x bounds.

        samples (`int`): number of 1D samples within the region of interest,
            total pixels will be samples^2.

        annotate_wvl (`bool`): whether to plot wavelength annotations.

        draw_plankian_locust (`bool`): whether to draw the plankian locust.

        fig (`matplotlib.figure.Figure`): figure to plot in.

        ax (`matplotlib.axes.Axis`): axis to plot in.

    Returns:
        `tuple` containing:

            `matplotlib.figure.Figure`: figure containing the plot.

            `matplotlib.axes.axis`: axis containing the plot.

    '''

    # duplicate xlim if ylim not set
    if ylim is None:
        ylim = xlim

    # don't compute over dead space
    xlim_bg = list(xlim)
    ylim_bg = list(ylim)
    if xlim[0] < 0:
        xlim_bg[0] = 0
    if xlim[1] > 0.65:
        xlim_bg[1] = 0.65
    if ylim[0] < 0:
        ylim_bg[0] = 0
    if ylim[1] > 0.6:
        ylim_bg[1] = 0.6

    # create lists of wavelengths and map them to uv for the border line and annotation.
    wvl_line = np.arange(400, 700, 2)
    wvl_line_uv = XYZ_to_uvprime(wavelength_to_XYZ(wvl_line))
    # duplicate the lowest wavelength so that the boundary line is closed
    wvl_line_uv = np.vstack((wvl_line_uv, wvl_line_uv[0, :]))

    background = render_cie_1976_background(*xlim_bg, *ylim_bg, samples)

    fig, ax = share_fig_ax(fig, ax)
    ax.imshow(background,
              extent=[*xlim_bg, *ylim_bg],
              interpolation='bilinear',
              origin='lower')
    ax.plot(wvl_line_uv[:, 0], wvl_line_uv[:, 1], ls='-', c='0.25', lw=2.5)
    if annotate_wvl:
        wvl_annotate = [360, 400, 455, 470, 480, 490,
                        500, 510, 520, 540, 555, 570, 580, 590,
                        600, 610, 625, 700, 830]
        fig, ax = cie_1976_wavelength_annotations(wvl_annotate, fig=fig, ax=ax)
    if draw_plankian_locust:
        fig, ax = cie_1976_plankian_locust(fig=fig, ax=ax)
    ax.set(xlim=xlim, xlabel='CIE u\'',
           ylim=ylim, ylabel='CIE v\'')

    return fig, ax


def cie_1931_wavelength_annotations(wavelengths, fig=None, ax=None):
    ''' Draws lines normal to the spectral locust on a CIE 1931 diagram and
        writes the text for each wavelength.

    Args:
        wavelengths (`iterable`): set of wavelengths to annotate.

        fig (`matplotlib.figure.Figure`): figure to draw on.

        ax (`matplotlib.axes.Axis`): axis to draw in.

    Returns:

        `tuple` containing:

            `matplotlib.figure.Figure`: figure containing the annotations.

            `matplotlib.axes.Axis`: axis containing the annotations.

    Notes:
        see SE:
        https://stackoverflow.com/questions/26768934/annotation-along-a-curve-in-matplotlib

    '''
    # some tick parameters
    tick_length = 0.025
    text_offset = 0.06

    # convert wavelength to u' v' coordinates
    wavelengths = np.asarray(wavelengths)
    idx = np.arange(1, len(wavelengths) - 1, dtype=int)
    wvl_lbl = wavelengths[idx]
    xy = XYZ_to_xy(wavelength_to_XYZ(wavelengths))
    x, y = xy[..., 0][idx], xy[..., 1][idx]
    x_last, y_last = xy[..., 0][idx - 1], xy[..., 1][idx - 1]
    x_next, y_next = xy[..., 0][idx + 1], xy[..., 1][idx + 1]

    angle = atan2(y_next - y_last, x_next - x_last) + pi / 2
    cos_ang, sin_ang = cos(angle), sin(angle)
    x1, y1 = x + tick_length * cos_ang, y + tick_length * sin_ang
    x2, y2 = x + text_offset * cos_ang, y + text_offset * sin_ang

    fig, ax = share_fig_ax(fig, ax)
    tick_lines = LineCollection(np.c_[x, y, x1, y1].reshape(-1, 2, 2), color='0.25', lw=1.25)
    ax.add_collection(tick_lines)
    for i in range(len(idx)):
        ax.text(x2[i], y2[i], str(wvl_lbl[i]), va="center", ha="center", clip_on=True)

    return fig, ax


def cie_1976_wavelength_annotations(wavelengths, fig=None, ax=None):
    ''' Draws lines normal to the spectral locust on a CIE 1976 diagram and
        writes the text for each wavelength.

    Args:
        wavelengths (`iterable`): set of wavelengths to annotate.

        fig (`matplotlib.figure.Figure`): figure to draw on.

        ax (`matplotlib.axes.Axis`): axis to draw in.

    Returns:

        `tuple` containing:

            `matplotlib.figure.Figure`: figure containing the annotations.

            `matplotlib.axes.Axis`: axis containing the annotations.

    Notes:
        see SE:
        https://stackoverflow.com/questions/26768934/annotation-along-a-curve-in-matplotlib

    '''
    # some tick parameters
    tick_length = 0.025
    text_offset = 0.06

    # convert wavelength to u' v' coordinates
    wavelengths = np.asarray(wavelengths)
    idx = np.arange(1, len(wavelengths) - 1, dtype=int)
    wvl_lbl = wavelengths[idx]
    uv = XYZ_to_uvprime(wavelength_to_XYZ(wavelengths))
    u, v = uv[..., 0][idx], uv[..., 1][idx]
    u_last, v_last = uv[..., 0][idx - 1], uv[..., 1][idx - 1]
    u_next, v_next = uv[..., 0][idx + 1], uv[..., 1][idx + 1]

    angle = atan2(v_next - v_last, u_next - u_last) + pi / 2
    cos_ang, sin_ang = cos(angle), sin(angle)
    u1, v1 = u + tick_length * cos_ang, v + tick_length * sin_ang
    u2, v2 = u + text_offset * cos_ang, v + text_offset * sin_ang

    fig, ax = share_fig_ax(fig, ax)
    tick_lines = LineCollection(np.c_[u, v, u1, v1].reshape(-1, 2, 2), color='0.25', lw=1.25)
    ax.add_collection(tick_lines)
    for i in range(len(idx)):
        ax.text(u2[i], v2[i], str(wvl_lbl[i]), va="center", ha="center", clip_on=True)

    return fig, ax


def cie_1976_plankian_locust(trange=(2000, 10000), num_points=100,
                             isotemperature_lines_at=None, isotemperature_du=0.025,
                             fig=None, ax=None):
    ''' draws the plankian locust on the CIE 1976 color diagram.

    Args:
        trange (`iterable`): (min,max) color temperatures.

        num_points (`int`): number of points to compute.

        isotemperature_lines_at (`iterable`): CCTs to plot isotemperature lines at,
            defaults to [2000, 3000, 4000, 5000, 6500, 10000] if None.
            set to False to not plot lines.

        isotemperature_du (`float`): delta-u, parameter, length in x of the isotemperature lines.

        fig (`matplotlib.figure.Figure`): figure to plot in.

        ax (`matplotlib.axes.Axis`): axis to plot in.

    Returns:
        `tuple` containing:

            `matplotlib.figure.Figure`. figure containing the plot.

            `matplotlib.axes.Axis`: axis containing the plot.

    '''
    cct = prepare_robertson_cct_data()
    cct_K, cct_u, cct_v, cct_dvdu = cct['K'], cct['u'], cct['v'], cct['dvdu']
    # compute the u', v' coordinates of the temperatures
    temps = np.linspace(trange[0], trange[1], num_points)
    interpf_u = interp1d(cct_K, cct_u)
    interpf_v = interp1d(cct_K, cct_v)
    u = interpf_u(temps)
    v = interpf_v(temps) * 1.5  # x1.5 converts 1960 uv to 1976 u' v'

    # if plotting isotemperature lines, compute the upper and lower points of
    # each line and connect them.
    plot_isotemp = True
    if isotemperature_lines_at is None:
        isotemperature_lines_at = np.asarray([2000, 3000, 4000, 5000, 6500, 10000])
        u_iso = interpf_u(isotemperature_lines_at)
        v_iso = interpf_v(isotemperature_lines_at)
        interpf_dvdu = interp1d(cct_u, cct_dvdu)

        dvdu = interpf_dvdu(u_iso)
        du = isotemperature_du / dvdu

        u_high = u_iso + du / 2
        u_low = u_iso - du / 2
        v_high = (v_iso + du / 2 * dvdu) * 1.5  # factors of 1.5 convert from uv to u'v'
        v_low = (v_iso - du / 2 * dvdu) * 1.5
    elif isotemperature_lines_at is False:
        plot_isotemp = False

    fig, ax = share_fig_ax(fig, ax)
    ax.plot(u, v, c='0.15')
    if plot_isotemp is True:
        for ul, uh, vl, vh in zip(u_low, u_high, v_low, v_high):
            ax.plot([ul, uh], [vl, vh], c='0.15')

    return fig, ax


@jit
def multi_cct_duv_to_upvp(cct, duv):
    upvp = np.empty((len(cct), len(duv), 2))
    for i, cct_v in enumerate(cct):
        for j, duv_v in enumerate(duv):
            values = CCT_Duv_to_uvprime(cct_v, duv_v)
            upvp[j, i, 0] = values[0]
            upvp[j, i, 1] = values[1]
    return upvp


def cct_duv_diagram(samples=100, fig=None, ax=None):
    ''' Creates a CCT-Duv diagram, for more information see Calculation of
        CCT and Duv and Practical Conversion Formulae, Yoshi Ohno, 2011.

    Args:
        samples (`int`): number of samples on the background.

        fig (`matplotlib.figure.Figure`): figure to plot in.

        ax (`matplotlib.axes.Axis`): axis to plot in.

    Returns:
        `tuple` containing:

            `matplotlib.figure.Figure`: figure containing the plot.

            `matplotlib.axes.Axis`: Axis containing the plot.

    '''
    # raise UserWarning('this type of plot is not yet properly implemented')
    xlim = (2000, 10000)
    ylim = (-0.03, 0.03)

    cct = np.linspace(xlim[0], xlim[1], samples)  # todo: even sampling along log, not linear
    duv = np.linspace(ylim[0], ylim[1], samples)

    upvp = multi_cct_duv_to_upvp(cct, duv)
    cct, duv = np.meshgrid(cct, duv)

    xy = uvprime_to_xy(upvp)
    xyz = xy_to_XYZ(xy)
    dat = XYZ_to_sRGB(xyz)

    maximum = np.max(dat, axis=-1)
    dat /= maximum[..., np.newaxis]
    dat = np.clip(dat, 0, 1)

    fig, ax = share_fig_ax(fig, ax)

    ax.imshow(dat,
              extent=[*xlim, *ylim],
              interpolation='bilinear',
              origin='lower',
              aspect='auto')

    ax.set(xlim=xlim, xlabel='CCT [K]',
           ylim=ylim, ylabel='Duv [a.u.]')

    return fig, ax


def spectrum_to_XYZ_emissive(spectrum_dict, cmf='1931_2deg'):
    ''' Converts a reflective or transmissive spectrum to XYZ coordinates.

    Args:
        spectrum_dict (`dict`): dictionary with wvl, values keys.  Wvl in units of nm.

        cmf (`str`): which color matching function to use, defaults to
            CIE 1931 2 degree observer.

    Returns:
        `tuple` containing:

            `float`: X

            `float`: Y

            `float`: Z

    '''
    wvl, values = spectrum_dict['wvl'], spectrum_dict['values']

    cmf = prepare_cmf(cmf)
    wvl_cmf = cmf['wvl']
    try:
        can_be_direct = np.allclose(wvl_cmf, wvl)
    except ValueError as e:
        can_be_direct = False
    if not can_be_direct:
        dat_interpf = interp1d(wvl, values, kind='linear', bounds_error=False, fill_value=0, assume_sorted=True)
        values = dat_interpf(wvl_cmf)

    dw = wvl_cmf[1] - wvl_cmf[0]
    k = 100 / (values * cmf['Y']).sum(axis=0) / dw
    X = k * (values * cmf['X']).sum(axis=0)
    Y = k * (values * cmf['Y']).sum(axis=0)
    Z = k * (values * cmf['Z']).sum(axis=0)
    return X, Y, Z


def spectrum_to_XYZ_nonemissive(spectrum_dict, illuminant='D65', cmf='1931_2deg'):
    ''' Converts a reflective or transmissive spectrum to XYZ coordinates.

    Args:
        spectrum_dict (`dict`): dictionary with wvl, values keys.  Wvl in units of nm.

        illuminant (`str`): reference illuminant, of the form "bb_[temperature]",
            or a CIE standard illuminant, e.g. D65, A, F1, etc.

        cmf (`str`): which color matching function to use, defaults to
            CIE 1931 2 degree observer.

    Returns:
        `tuple` containing:

            `float`: X

            `float`: Y

            `float`: Z

    '''
    wvl, values = spectrum_dict['wvl'], spectrum_dict['values']

    cmf = prepare_cmf(cmf)
    wvl_cmf = cmf['wvl']
    try:
        can_be_direct = np.allclose(wvl_cmf, wvl)
    except ValueError as e:
        can_be_direct = False

    if not can_be_direct:
        dat_interpf = interp1d(wvl, values, kind='linear', bounds_error=False, fill_value=0, assume_sorted=True)
        values = dat_interpf(wvl_cmf)

    ill_spectrum = prepare_illuminant_spectrum(illuminant)

    try:
        can_be_direct_illuminant = np.allclose(wvl_cmf, ill_spectrum['wvl'])
    except ValueError as e:
        can_be_direct_illuminant = False
    if can_be_direct_illuminant:
        ill_spectrum = ill_spectrum['values']
    else:
        ill_wvl, ill_vals = ill_spectrum['wvl'], ill_spectrum['values']
        ill_interpf = interp1d(ill_wvl, ill_vals, kind='linear', bounds_error=False, fill_value=0, assume_sorted=True)
        ill_spectrum = ill_interpf(wvl_cmf)

    dw = wvl_cmf[1] - wvl_cmf[0]
    k = 100 / (values * ill_spectrum * cmf['Y']).sum() / dw
    X = k * (values * ill_spectrum * cmf['X']).sum()
    Y = k * (values * ill_spectrum * cmf['Y']).sum()
    Z = k * (values * ill_spectrum * cmf['Z']).sum()
    return X, Y, Z


def wavelength_to_XYZ(wavelength, observer='1931_2deg'):
    ''' Uses tristimulus color matching functions to map a awvelength to XYZ
        coordinates.

    Args:
        wavelength (`float`): wavelength in nm.

        observer (`str`): CIE observer name, must be 1931_2deg.

    Returns:
        `numpy.ndarray`: array with last dimension corresponding to X, Y, Z.

    '''
    wavelength = np.asarray(wavelength, dtype=config.precision)

    cmf = prepare_cmf(observer)
    wvl, X, Y, Z = cmf['wvl'], cmf['X'], cmf['Y'], cmf['Z']

    ia = {'bounds_error': False, 'fill_value': 0, 'assume_sorted': True}
    f_X, f_Y, f_Z = interp1d(wvl, X, **ia), interp1d(wvl, Y, **ia), interp1d(wvl, Z, **ia)
    x, y, z = f_X(wavelength), f_Y(wavelength), f_Z(wavelength)

    shape = wavelength.shape
    return np.stack((x, y, z), axis=len(shape))


def XYZ_to_xyY(XYZ, assume_nozeros=True, ref_white='D65'):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

        assume_nozeros (`bool`): assume there are no zeros present, computation
            will run faster as `True`, if `False` will correct for all-zero
            values.

        ref_white (`str): string for reference illuminant used in the case
            where X==Y==Z==0.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

            `numpy.ndarray`: Y coordinates.

    Notes:
        If X==Y==Z==0 and assume_nozeros is False, will return the chromaticity coordinates
        of the reference white.

    '''
    XYZ = np.asarray(XYZ)
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]

    if not assume_nozeros:
        zero_X = X == 0
        zero_Y = Y == 0
        zero_Z = Z == 0
        allzeros = np.all(np.dstack((zero_X, zero_Y, zero_Z)))
        X[allzeros] = 0.3
        Y[allzeros] = 0.3
        Z[allzeros] = 0.3

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    Y = Y
    shape = x.shape

    if not assume_nozeros:
        spectrum = prepare_illuminant_spectrum(ref_white)
        xyz = spectrum_to_XYZ_emissive(spectrum)
        xr, yr = XYZ_to_xy(xyz)
        x[allzeros] = xr
        y[allzeros] = yr
        Y[:] = xyz[1]

    return np.stack((x, y, Y), axis=len(shape))


def XYZ_to_xy(XYZ):
    ''' Converts XYZ points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

    '''
    xyY = XYZ_to_xyY(XYZ)
    return xyY_to_xy(xyY)


def XYZ_to_uvprime(XYZ):
    ''' Converts XYZ points to u'v' points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: u' coordinates.

            `numpy.ndarray`: u' coordinates.

    '''
    XYZ = np.asarray(XYZ)
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    u = (4 * X) / (X + 15 * Y + 3 * Z)
    v = (9 * Y) / (X + 15 * Y + 3 * Z)

    shape = u.shape
    return np.stack((u, v), axis=len(shape))


def xyY_to_xy(xyY):
    ''' converts xyY points to xy points.

    Args:
        xyY (`numpy.ndarray`): ndarray with last dimension corresponding to
            x, y, Y.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

    '''
    xyY = np.asarray(xyY)
    shape = xyY.shape
    if shape[-1] is 2:
        return xyY
    else:
        x, y = xyY[..., 0], xyY[..., 1]

        shape = x.shape
        return np.stack((x, y), axis=len(shape))


def xyY_to_XYZ(xyY):
    ''' converts xyY points to XYZ points.

    Args:
        xyY (`numpy.ndarray`): ndarray with last dimension corresponding to
            x, y, Y.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: X coordinates.

            `numpy.ndarray`: Y coordinates.

            `numpy.ndarray`: Z coordinates.

    '''
    xyY = np.asarray(xyY)
    x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
    y_l = y.copy()
    idxs = y_l == 0
    y_l[idxs] = 0.3
    X = np.asarray((x * Y) / y_l)
    Y = np.asarray(Y)
    Z = np.asarray(((1 - x - y_l) * Y) / y_l)
    X[idxs] = 0
    Y[idxs] = 0
    Z[idxs] = 0

    shape = X.shape
    return np.stack((X, Y, Z), axis=len(shape))


def xy_to_xyY(xy, Y=1):
    ''' converts xy points to xyY points.

    Args:
        xy (`numpy.ndarray`): ndarray with last dimension corresponding to
            x, y.

        Y (`numpy.ndarray`): Y value to fill with.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

            `numpy.ndarray`: Y coordinates.

    '''
    xy = np.asarray(xy)
    shape = xy.shape
    if shape[-1] is 3:
        return xy
    else:
        x, y = xy[..., 0], xy[..., 1]
        Y = np.ones(x.shape) * Y

        return np.stack((x, y, Y), axis=len(shape) - 1)


def xy_to_XYZ(xy):
    ''' converts xy points to xyY points.

    Args:
        xy (`numpy.ndarray`): ndarray with last dimension corresponding to
            x, y.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: X coordinates.

            `numpy.ndarray`: Y coordinates.

            `numpy.ndarray`: Z coordinates.

    '''
    xy = np.asarray(xy)
    xyY = xy_to_xyY(xy)
    return xyY_to_XYZ(xyY)


def xy_to_CCT(xy):
    ''' Computes the correlated color temperature given x,y chromaticity coordinates.

    Args:
        xy (`iterable`): x, y chromaticity coordinates.

    Returns:
        `float`: CCT.

    '''
    xy = np.asarray(xy)
    x, y = xy[..., 0], xy[..., 1]
    n = (x - 0.3320) / (0.1858 - y)
    return 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.3


def uvprime_to_xy(uvprime):
    ''' Converts u' v' points to xyY x,y points.

    Args:
        uv (`numpy.ndarray`): ndarray with last dimension corresponding to
            u', v'.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

    '''
    uv = np.asarray(uvprime)
    u, v = uv[..., 0], uv[..., 1]
    x = (9 * u) / (6 * u - 16 * v + 12)
    y = (4 * v) / (6 * u - 16 * v + 12)

    shape = x.shape
    return np.stack((x, y), axis=len(shape))


def uvprime_to_CCT(uvprime):
    ''' Computes CCT from u'v' coordinates.

    Args:
        uv (`numpy.ndarray`): array with last dimensions corresponding to u, v

    Returns:
        `float`: CCT.

    '''
    uv = np.asarray(uvprime)
    xy = uvprime_to_xy(uv)
    return xy_to_CCT(xy)


def uvprime_to_Duv(uvprime):
    ''' Computes Duv from u'v' coordiantes.

    Args:
        uv (`numpy.ndarray`): array with last dimensions corresponding to u, v

    Returns:
        `float`: CCT.

    Notes:
        see "Calculation of CCT and Duv and Practical Conversion Formulae", Yoshi Ohno
        http://www.cormusa.org/uploads/CORM_2011_Calculation_of_CCT_and_Duv_and_Practical_Conversion_Formulae.PDF
    '''
    k0, k1, k2, k3 = CIE_DUV_k0, CIE_DUV_k1, CIE_DUV_k2, CIE_DUV_k3
    k4, k5, k6 = CIE_DUV_k4, CIE_DUV_k5, CIE_DUV_k6

    uv = np.asarray(uvprime)
    u, v = uv[..., 0], uv[..., 1] / 1.5  # inline convert v' to v
    L_FP = sqrt((u - 0.292) ** 2 + (v - 0.24) ** 2)
    a = arccos((u - 0.292) / L_FP)
    L_BB = k6 * a ** 6 + k5 * a ** 5 + k4 * a ** 4 + k3 * a ** 3 + k2 * a ** 2 + k1 * a + k0
    return L_FP - L_BB


def uvprime_to_CCT_Duv(uvprime):
    ''' Computes CCT and Duv from u'v' coordiantes.

    Args:
        uv (`numpy.ndarray`): array with last dimensions corresponding to u, v

    Returns:
        `float`: CCT.

    Notes:
        see "Calculation of CCT and Duv and Practical Conversion Formulae", Yoshi Ohno
        http://www.cormusa.org/uploads/CORM_2011_Calculation_of_CCT_and_Duv_and_Practical_Conversion_Formulae.PDF

    '''
    duv = uvprime_to_Duv(uvprime)
    cct = uvprime_to_CCT(uvprime)
    return cct, duv


def CCT_Duv_to_uvprime(CCT, Duv, delta_t=0.01):
    ''' Converts (CCT,Duv) coordinates to upvp coordinates.

    Args:
        CCT (`float` or `iterable`): CCT coordinate.

        Duv (`float` or `iterable`): Duv coordinate.

        delta_t (`float`): temperature differential used to compute the tangent
            line to the plankian locust.  Default to 0.01, Ohno suggested (2011).

    Returns:
        `tuple` containing:

            `float` u'

            `float` v'

    '''
    CCT, Duv = np.asarray(CCT), np.asarray(Duv)

    wvl = np.arange(360, 835, 5)
    bb_spec_0 = blackbody_spectrum(CCT, wvl)
    bb_spec_1 = blackbody_spectrum(CCT + delta_t, wvl)
    bb_spec_0 = {
        'wvl': wvl,
        'values': bb_spec_0,
    }
    bb_spec_1 = {
        'wvl': wvl,
        'values': bb_spec_1,
    }

    xyz_0 = spectrum_to_XYZ_emissive(bb_spec_0)
    xyz_1 = spectrum_to_XYZ_emissive(bb_spec_1)
    upvp_0 = XYZ_to_uvprime(xyz_0)
    upvp_1 = XYZ_to_uvprime(xyz_1)

    u0, v0 = upvp_0[..., 0], upvp_0[..., 1]
    u1, v1 = upvp_1[..., 0], upvp_1[..., 1]
    du, dv = u1 - u0, v1 - v0
    u = u0 + Duv * dv / sqrt(du**2 + dv**2)
    v = u0 + Duv * du / sqrt(du**2 + dv**2)
    return u, v * 1.5**2  # factor of 1.5 converts v -> v'


def spectrum_to_CCT_Duv(spectrum_dict):
    ''' Computes the CCT and Duv values of a spectrum object.

    Args:
        spectrum_dict (`dict`): dictionary with keys wvl, values.

    Returns:
        `tuple` containing (CCT, Duv)

    '''
    XYZ = spectrum_to_XYZ_nonemissive(spectrum_dict)
    upvp = XYZ_to_uvprime(XYZ)
    CCT = uvprime_to_CCT(upvp)
    Duv = uvprime_to_Duv(upvp)
    return (CCT, Duv)


def XYZ_to_AdobeRGB(XYZ, illuminant='D65'):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

        illuminant (`str`): which illuminant to use, either D65 or D50.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: R coordinates.

            `numpy.ndarray`: G coordinates.

            `numpy.ndarray`: B coordinates.

    '''
    if illuminant.upper() == 'D65':
        invmat = COLOR_MATRICIES['AdobeRGB']['D65']
    elif illuminant.upper() == 'D50':
        invmat = COLOR_MATRICIES['AdobeRGB']['D50']
    else:
        raise ValueError('Must use D65 or D50 illuminant.')

    return XYZ_to_RGB(XYZ, invmat)


def XYZ_to_sRGB(XYZ, illuminant='D65', gamma_encode=True):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

        illuminant (`str`): which illuminant to use, either D65 or D50.

        gamma_encode (`bool`): if True, apply sRGB_oetf to the data for display,
            if false, leave values in linear regime.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: R coordinates.

            `numpy.ndarray`: G coordinates.

            `numpy.ndarray`: B coordinates.

    Notes:
        Returns are linear, need to be raised to 2.4 power to make correct
            values for viewing.

    '''
    if illuminant.upper() == 'D65':
        invmat = COLOR_MATRICIES['sRGB']['D65']
    elif illuminant.upper() == 'D50':
        invmat = COLOR_MATRICIES['sRGB']['D50']
    else:
        raise ValueError('Must use D65 or D50 illuminant.')

    if gamma_encode is True:
        rgb = XYZ_to_RGB(XYZ, invmat)
        return sRGB_oetf(rgb)
    else:
        return XYZ_to_RGB(XYZ, invmat)


def XYZ_to_RGB(XYZ, conversion_matrix, XYZ_scale=100):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

        conversion_matrix (`str`): conversion matrix to use to convert XYZ
            to RGB values.

        XYZ_scale (`float`): maximum value of XYZ values; XYZ will be normalized
            by this prior to conversion.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: R coordinates.

            `numpy.ndarray`: G coordinates.

            `numpy.ndarray`: B coordinates.

    '''
    XYZ = np.asarray(XYZ) / XYZ_scale
    if len(XYZ.shape) == 1:
        return np.matmul(conversion_matrix, XYZ)
    else:
        return np.tensordot(XYZ, conversion_matrix, axes=((2), (1)))


def sRGB_oetf(L):
    ''' sRGB opto-electrical transfer function.  Similar to gamma.

    Args:
        L (`numpy.ndarray`): sRGB values.

    Returns:
        `numpy.ndarray`: L', L modulated by the oetf.

    Notes:
        input must be an array, cannot be a scalar.
    '''
    L = np.asarray(L)
    negative = L < 0
    L_l = L.copy()
    L_l[negative] = 0.0
    return np.where(L_l <= 0.0031308, L_l * 12.92, 1.055 * (L_l ** (1 / 2.4)) - 0.055)


def sRGB_reverse_oetf(V):
    ''' sRGB opto-electrical transfer function.  Similar to gamma.

    Args:
        V (`numpy.ndarray`): sRGB values.

    Returns:
        `numpy.ndarray`: V', V modulated by the oetf.

    '''
    V = np.asarray(V)
    return np.where(V <= sRGB_oetf(0.0031308), V / 12.92, ((V + 0.055) / 1.055) ** 2.4)
