import logging
from typing import Tuple, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.fftpack import next_fast_len

from dexpv2.utils import center_crop, pad_to_shape

LOG = logging.getLogger(__name__)


def phase_cross_corr(
    ref_img: ArrayLike, mov_img: ArrayLike, maximum_shift: float = 1.0
) -> Tuple[int, ...]:
    """
    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.

    Parameters
    ----------
    ref_img : ArrayLike
        Reference image.
    mov_img : ArrayLike
        Moved image.
    maximum_shift : float, optional
        Maximum location shift normalized by axis size, by default 1.0

    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """
    shape = tuple(
        cast(int, next_fast_len(int(max(s1, s2) * maximum_shift)))
        for s1, s2 in zip(ref_img.shape, mov_img.shape)
    )

    LOG.info(
        f"phase cross corr. fft shape of {shape} for arrays of shape {ref_img.shape} and {mov_img.shape} "
        f"with maximum shift of {maximum_shift}"
    )

    if np.any(shape > ref_img.shape):
        padded_shape = np.maximum(ref_img, shape)
        ref_img = pad_to_shape(ref_img, padded_shape, mode="reflect")
        mov_img = pad_to_shape(mov_img, padded_shape, mode="reflect")

    if np.any(shape < ref_img.shape):
        ref_img = center_crop(ref_img, shape)
        mov_img = center_crop(mov_img, shape)

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps

    norm = np.fmax(np.abs(Fimg1) * np.abs(Fimg2), eps)
    corr = np.fft.irfftn(Fimg1 * Fimg2.conj() / norm)
    corr = np.fft.fftshift(np.abs(corr))

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))

    LOG.info(f"phase cross corr. peak at {peak}")

    return peak
