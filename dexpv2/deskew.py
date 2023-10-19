import math

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm


def deskewing_shift(
    angle: float,
    z_step: float,
    x_res: float,
) -> float:
    """
    Calculate the shift required for deskewing.

    Parameters
    ----------
    angle : float
        The angle in degrees to be deskewed.
    z_step : float
        The z-step size.
    x_res : float
        The x-axis resolution in the units as `z_step`.

    Returns
    -------
    float
        The calculated shift.
    """
    return np.cos(np.deg2rad(angle)) * z_step / x_res


def get_deskewed_shape(
    shape: tuple,
    shift: float,
) -> tuple:
    """
    Calculate the new shape after deskewing.

    Parameters
    ----------
    shape : tuple
        The original shape of the array (z, y, x).
    shift : float
        The calculated shift for deskewing.

    Returns
    -------
    tuple
        The new shape after deskewing.
    """
    assert len(shape) == 3

    offset = int(math.fabs(shift) * shape[0] + 0.5)

    shape_list = list(shape)
    shape_list[-1] += offset

    return tuple(shape_list)


def deskew(
    raw_data: ArrayLike,
    shift: float,
) -> ArrayLike:
    """
    Perform the deskewing operation on the raw data.

    Parameters
    ----------
    raw_data : ArrayLike
        The raw 3D data array with shape (z, y, x).
    shift : float
        The calculated shift for deskewing.

    Returns
    -------
    ArrayLike
        The deskewed 3D data array.
    """
    new_shape = get_deskewed_shape(raw_data.shape, shift)

    offset = new_shape[-1] - raw_data.shape[-1]
    out_data = np.zeros(new_shape, dtype=raw_data.dtype, like=raw_data)

    x_length = raw_data.shape[-1] - 1
    w1 = shift - math.floor(shift)
    w2 = math.ceil(shift) - shift

    if shift > 0:
        offset = 0

    for z in tqdm(range(new_shape[0])):
        current_shift = int(math.floor(z * shift)) + offset
        out_data[z, :, current_shift : current_shift + x_length] = (
            w1 * raw_data[z, :, :-1] + w2 * raw_data[z, :, 1:]
        )

    return out_data
