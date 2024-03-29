import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

LOG = logging.getLogger(__name__)


def center_crop(arr: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
    """Crops the center of `arr`"""
    assert arr.ndim == len(shape)

    starts = tuple((cur_s - s) // 2 for cur_s, s in zip(arr.shape, shape))

    assert all(s >= 0 for s in starts)

    slicing = tuple(slice(s, s + d) for s, d in zip(starts, shape))

    LOG.info(
        f"center crop: input shape {arr.shape}, output shape {shape}, slicing {slicing}"
    )

    return arr[slicing]


def pad_to_shape(
    arr: ArrayLike, shape: Tuple[int, ...], mode: str, **kwargs
) -> ArrayLike:
    """Pads array to shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int]
        Output shape.
    mode : str
        Padding mode (see np.pad).

    Returns
    -------
    ArrayLike
        Padded array.
    """
    assert arr.ndim == len(shape)

    dif = tuple(s - a for s, a in zip(shape, arr.shape))
    assert all(d >= 0 for d in dif)

    pad_width = [[s // 2, s - s // 2] for s in dif]

    LOG.info(
        f"padding: input shape {arr.shape}, output shape {shape}, padding {pad_width}"
    )

    return np.pad(arr, pad_width=pad_width, mode=mode, **kwargs)


def to_cpu(arr: ArrayLike) -> ArrayLike:
    """Moves array to cpu, if it's already there nothing is done."""
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    elif hasattr(arr, "get"):
        arr = arr.get()
    return arr


def translation_slicing(translation: ArrayLike) -> Tuple[slice, ...]:
    """
    Helper function to get translation slicing from negative or positive integers for translation.

    For example:
    (1, -2) -> (slice(1, None), slice(None, -2))

    Parameters
    ----------
    translation : ArrayLike
        Array of translations.

    Returns
    -------
    Tuple[slice, ...]
        Slicing equivalent to translation.
    """
    return tuple(slice(None, s) if s < 0 else slice(s, None) for s in translation)


def normalize(arr: ArrayLike, dtype: Optional[np.dtype] = None) -> ArrayLike:
    """Normalizes array into the range of (0, 1)."""
    if dtype:
        arr = arr.astype(dtype)
    arr = arr - arr.min()
    return arr / arr.max()


def standardize(arr: ArrayLike, dtype: Optional[np.dtype] = None) -> ArrayLike:
    """Standardizes array into mean zero and standard deviation one."""
    if dtype:
        arr = arr.astype(dtype)
    arr = arr - arr.mean()
    return arr / arr.std()
