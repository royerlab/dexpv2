from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.utils import translation_slicing


def translated_views(
    img: ArrayLike,
    translation: ArrayLike,
    perturb_half: bool = False,
    **kwargs,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Generates translated views of the input image.
    Additional perturbation of opposite halves are possible to simulate a lightsheet dataset.

    Parameters
    ----------
    img : ArrayLike
        Input image.
    translation : ArrayLike
        Translation between views.
    perturb_half : bool, optional
        When true opposite halfs of the image are perturbed, by default False
    kwargs :
        linear_blending keyword arguments

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Tuple of two views of the original image displaced by `translation`.
    """

    img1 = img[translation_slicing(translation)]
    img2 = img[translation_slicing(-translation)]

    assert img1.shape == img2.shape

    if perturb_half:
        blending = linear_blending(img1.shape[-1], **kwargs)
        blending = np.asarray(blending, like=img)
        img1 = img1 * blending[None, None, ...]
        img2 = img2 * np.flip(blending)[None, None, ...]

    return img1, img2


def linear_blending(length: int, x_range: float, baseline: float) -> ArrayLike:
    """
    Creates a linear blending map of `length` from (-`x_range`, `x_range`)
    using a sigmoid function given a range and a minimum baseline value.

    Parameters
    ----------
    length : int
        Length of blending map.
    x_range : float
        Range of sigmoid input x.
    baseline : float
        Baseline minimum value.

    Returns
    -------
    ArrayLike
        Blending map.
    """
    x = np.linspace(-x_range, x_range, length)
    y = 1 / (1 + np.exp(-x))
    y = (y + baseline) / (1 + baseline)
    return y
