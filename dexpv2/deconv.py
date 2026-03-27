import logging
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.fftpack import next_fast_len

from dexpv2.utils import center_crop, pad_to_shape

LOG = logging.getLogger(__name__)


def _ensure_nonnegative(arr: ArrayLike, zero: float = 0.0) -> ArrayLike:
    """Makes the input array non-negative."""
    return np.fmax(arr, zero, out=arr)


def _preprocess(img: ArrayLike, psf: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """Checks inputs data type and resizes for faster FFT."""
    if not np.issubdtype(img.dtype, np.floating):
        raise ValueError(f"`img` must be `float`. Found {img.dtype}.")

    LOG.info(f"`img` dtype = {img.dtype}")

    new_shape = tuple(next_fast_len(s) for s in img.shape)

    LOG.info(f"FFT shape {new_shape} from {img.shape}")

    psf = psf.astype(img.dtype)
    psf = psf / psf.sum()

    img = pad_to_shape(img, shape=new_shape, mode="constant")
    psf = pad_to_shape(psf, shape=new_shape, mode="constant")

    assert img.shape == psf.shape

    return img, psf


def _lucy_richardson(
    img: ArrayLike,
    otf: ArrayLike,
    iterations: int,
) -> ArrayLike:
    """
    Standard Lucy-Richardson deconvolution.
    Lower memory usage than accelerated deconvolution.
    """
    eps = np.finfo(img.dtype).eps

    y = img
    for i in range(iterations):
        reblur = np.fft.irfftn(otf * np.fft.rfftn(y), img.shape)
        reblur = _ensure_nonnegative(reblur, eps)
        im_ratio = img / reblur
        estimate = np.fft.irfftn(np.conj(otf) * np.fft.rfftn(im_ratio), img.shape)
        del reblur, im_ratio

        estimate = _ensure_nonnegative(estimate)
        y = y * estimate
        del estimate

    return y


def _accelerated_lucy_richardson(
    img: ArrayLike,
    otf: ArrayLike,
    iterations: int,
) -> ArrayLike:
    """Accelerated Lucy-Richardson deconvolution using Andrew-Biggs step scaling."""

    eps = np.finfo(img.dtype).eps

    t = img
    tm1 = np.zeros_like(t)
    g_tm1 = np.zeros_like(t)
    g_tm2 = np.zeros_like(t)

    for i in range(iterations):
        # first order andrew biggs acceleration
        # acceleration is proportional to the dot product between `g_tm1` and `g_tm2`
        alpha = (g_tm1 * g_tm2).sum() / (np.square(g_tm2).sum() + eps)
        alpha = np.clip(alpha, 0, 1)
        h1_t = t - tm1
        y = t + alpha * h1_t

        t = _ensure_nonnegative(y)

        # update
        reblur = np.fft.irfftn(otf * np.fft.rfftn(t), img.shape)
        reblur = _ensure_nonnegative(reblur, eps)
        im_ratio = img / reblur
        estimate = np.fft.irfftn(np.conj(otf) * np.fft.rfftn(im_ratio), img.shape)

        estimate = _ensure_nonnegative(estimate)
        tp1 = t * estimate
        # update g's
        g_tm2 = g_tm1
        # this is where the magic is, we need to compute from previous step
        # which may have been augmented by acceleration
        g_tm1 = tp1 - y
        t, tm1 = tp1, t

    return t


def lucy_richardson(
    img: ArrayLike,
    psf: ArrayLike,
    iterations: int,
    accelerated: bool = True,
) -> ArrayLike:
    """
    Lucy-Richardson deconvolution.

    Modified from:
    https://github.com/david-hoffman/pydecon/blob/master/CUPY/decon_onefile.py
    Apache License 2.0 from David Hoffman

    Andrew-Biggs accelerated restoration reference:
    http://ischebeck.net/dr/bibliography/pdf/biggs97a.pdf

    Parameters
    ----------
    img : ArrayLike
        Image in a floating point representation.
    psf : ArrayLike
        Point spread function array.
    iterations : int
        Number of iterations.
    accelerated : bool
        Flag indicating Andrew-Biggs accelerated restoration.

    Returns
    -------
    ArrayLike
        Deconvolved image.
    """
    orig_shape = img.shape
    img, psf = _preprocess(img, psf)

    otf = np.fft.rfftn(np.fft.ifftshift(psf))
    del psf

    if accelerated:
        img = _accelerated_lucy_richardson(img, otf, iterations)
    else:
        img = _lucy_richardson(img, otf, iterations)

    return center_crop(img, orig_shape)
