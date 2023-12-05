import logging
from typing import Callable, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.fftpack import next_fast_len

from dexpv2.constants import DEXPV2_DEBUG
from dexpv2.utils import center_crop, pad_to_shape, to_cpu

LOG = logging.getLogger(__name__)


def _match_shape(img: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
    """Pad or crop array to match provided shape."""

    if np.any(shape > img.shape):
        padded_shape = np.maximum(img.shape, shape)
        img = pad_to_shape(img, padded_shape, mode="reflect")

    if np.any(shape < img.shape):
        img = center_crop(img, shape)

    return img


def phase_cross_corr(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    maximum_shift: float = 1.0,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
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

    ref_img = _match_shape(ref_img, shape)
    mov_img = _match_shape(mov_img, shape)

    ref_img = to_device(ref_img)
    mov_img = to_device(mov_img)

    ref_img = np.log1p(ref_img)
    mov_img = np.log1p(mov_img)

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()
    del Fimg1, Fimg2

    norm = np.fmax(np.abs(prod), eps)
    corr = np.fft.irfftn(prod / norm)
    del prod, norm

    corr = np.fft.fftshift(np.abs(corr))

    argmax = to_cpu(np.argmax(corr))
    peak = np.unravel_index(argmax, corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))

    LOG.info(f"phase cross corr. peak at {peak}")

    if DEXPV2_DEBUG:
        import napari

        napari.view_image(to_cpu(np.square(corr).real))
        napari.run()

    return peak


def multiview_phase_cross_corr(
    C0L0: ArrayLike,
    C0L1: ArrayLike,
    C1L0: ArrayLike,
    C1L1: ArrayLike,
    camera_1_flip: bool,
    maximum_shift: float = 1.0,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
) -> Tuple[int, ...]:
    """
    Computes the translation between cameras.

    Parameters
    ----------
    C0L0 : ArrayLike
        View from camera 0 and light sheet 0.
    C0L1 : ArrayLike
        View from camera 0 and light sheet 0.
    C1L0 : ArrayLike
        View from camera 1 and light sheet 0.
    C1L1 : ArrayLike
        View from camera 1 and light sheet 1.
    camera_1_flip : ArrayLike
        Indicates if camera 1 is flipped on the last axis.
    maximum_shift : float, optional
        Maximum location shift normalized by axis size, by default 0.1
    to_device : Callable, optional
        Helper function to send data to specialized device, this function sends the data
        to the device only when needed, reducing the memory usage.

    Returns
    -------
    ArrayLike
        Translation between cameras.
    """
    camera_0 = C0L0.astype(np.float32) + C0L1
    camera_1 = C1L0.astype(np.float32) + C1L1

    if camera_1_flip:
        camera_1 = np.flip(camera_1, -1)

    return phase_cross_corr(
        camera_0,
        camera_1,
        maximum_shift=maximum_shift,
        to_device=to_device,
    )
