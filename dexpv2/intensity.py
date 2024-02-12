from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


def estimate_quantiles(
    arr: ArrayLike,
    lower: float = 0.01,
    upper: float = 0.9999,
    downsampling: Optional[int] = 4,
) -> Tuple[float, float]:
    """
    Estimate quantiles from an array.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    lower : float
        Lower quantile.
    upper : float
        Upper quantile.
    downsampling : int
        Downsampling factor, no downsampling if None.

    Returns
    -------
    Tuple[float, float]
        Lower and upper quantiles.
    """

    if downsampling is not None:
        slicing = tuple(slice(None, None, downsampling) for _ in arr.shape)
        arr = arr[slicing]

    lower_quantile = np.quantile(arr, lower)
    upper_quantile = np.quantile(arr, upper)

    return lower_quantile, upper_quantile


def equalize_views(
    view_0: ArrayLike,
    view_1: ArrayLike,
    **kwargs,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Equalize the intensity of two views to the brightest view, INPLACE.

    Parameters
    ----------
    view_0 : ArrayLike
        View 0.
    view_1 : ArrayLike
        View 1.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Equalized views.
    """

    lower_0, upper_0 = estimate_quantiles(view_0, **kwargs)
    lower_1, upper_1 = estimate_quantiles(view_1, **kwargs)

    if upper_1 > upper_0:
        factor = (upper_1 - lower_1) / (upper_0 - lower_0)
        np.subtract(view_0, lower_0, out=view_0, casting="unsafe")
        np.multiply(view_0, factor, out=view_0, casting="unsafe")
        np.add(view_0, lower_1, out=view_0, casting="unsafe")
    else:
        factor = (upper_0 - lower_0) / (upper_1 - lower_1)
        np.subtract(view_1, lower_1, out=view_1, casting="unsafe")
        np.multiply(view_1, factor, out=view_1, casting="unsafe")
        np.add(view_1, lower_0, out=view_1, casting="unsafe")

    return view_0, view_1
