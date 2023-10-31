import math

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm


def deskewing_shift(
    angle: float,
    scan_step: float,
    pixel_size: float,
) -> float:
    """
    Calculate the shift required for deskewing.

    TILT AXIS
        /
       /
      /
     /
    / o ANGLE
    -------------
      SCAN AXIS

    Parameters
    ----------
    angle : float
        The angle in degrees to be deskewed.
    scan_step : float
        The scan step size along tile axis.
    pixel_size : float
        The pixel size (resolution).

    Returns
    -------
    float
        The calculated shift.
    """
    return np.cos(np.deg2rad(angle)) * pixel_size / scan_step


def deskewed_dimension(
    angle: float,
    scan_step: float,
) -> float:
    """
    Calculate the length of the deskewed image (z-axis) opposite to angle.

    Parameters
    ----------
    angle : float
        The angle in degrees to be deskewed.
    scan_step : float
        The scan_step size.

    Returns
    -------
    float
        The calculated length.
    """
    return np.sin(np.deg2rad(angle)) * scan_step


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
    if len(shape) != 3:
        raise ValueError(f"The shape must be 3D. Got {shape}.")

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
    Skewing is performed along the xz-slice.

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
    out_data = np.zeros(new_shape, dtype=raw_data.dtype, like=raw_data)

    x_length = raw_data.shape[-1] - 1

    if shift > 0:
        offset = 0
    else:
        offset = new_shape[-1] - raw_data.shape[-1]

    shift_per_z = np.arange(new_shape[0]) * shift + offset

    w_left = np.ceil(shift_per_z) - shift_per_z
    w_right = 1.0 - w_left

    x_length = raw_data.shape[-1] - 1
    int_shift = np.floor(shift_per_z).astype(int)

    if shift > 0:
        # this asserts that the interpolation is centered at the pixels values
        int_shift += 1
        w_left, w_right = w_right, w_left

    for z in tqdm(range(new_shape[0])):
        current_shift = int_shift[z]
        out_data[z, :, current_shift : current_shift + x_length] = (
            w_left[z] * raw_data[z, :, :-1] + w_right[z] * raw_data[z, :, 1:]
        )

    return out_data
