from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.utils import translation_slicing


def translated_views(
    img: ArrayLike,
    translation: ArrayLike,
    perturb_along_axis: Optional[int] = None,
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
    perturb_along_axis : bool, optional
        When supplied perturb array with blending along axis, by default False
    kwargs :
        linear_blending keyword arguments when perturb half is True.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Tuple of two views of the original image displaced by `translation`.
    """

    img1 = img[translation_slicing(translation)]
    img2 = img[translation_slicing(-translation)]

    assert img1.shape == img2.shape

    if perturb_along_axis is not None:
        length = img1.shape[perturb_along_axis]
        shape = np.ones(img1.ndim, dtype=int)
        shape[perturb_along_axis] = length

        blending = linear_blending(length, **kwargs)
        blending = np.asarray(blending, like=img).reshape(shape)

        img1 = img1 * blending
        img2 = img2 * np.flip(blending, axis=perturb_along_axis)

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
