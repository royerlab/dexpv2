import math
from typing import Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.cuda import import_module, to_numpy


def _scanning_and_scale(
    pixel_size: float,
    scan_step: float,
    tilt_angle: float,
) -> Tuple[int, float]:
    """Compute the resample factor and lateral scaling for deskewing."""
    x_res = pixel_size * math.cos(math.radians(tilt_angle))
    resample_factor = int(round(scan_step / x_res))
    lateral_scaling = math.cos(math.radians(tilt_angle))
    return resample_factor, lateral_scaling


def get_deskewed_shape(
    shape: tuple,
    pixel_size: float,
    scan_step: float,
    tilt_angle: float,
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

    resample_factor, lateral_scaling = _scanning_and_scale(
        pixel_size, scan_step, tilt_angle
    )

    x_size = int(round((shape[0] * resample_factor + shape[2]) * lateral_scaling))

    out_shape = (
        math.ceil(shape[2] / resample_factor) - 1,
        shape[1],
        x_size,
    )

    return out_shape


def _deskew(
    crop: ArrayLike,
    resample_factor: int,
    lateral_scaling: float,
) -> ArrayLike:
    """
    Bin Yang deskew.
    Deskew a volume with an angle between the x and z axis.
    Reference: https://github.com/royerlab/dexp/blob/master/dexp/processing/deskew/yang_deskew.py

    Parameters
    ----------
    crop : ArrayLike
        The volume to be deskewed.
    resample_factor : int
        Step size for resampling (deskewing).
    lateral_scaling : float
        The pixel size lateral scaling.

    Returns
    -------
    ArrayLike
        The deskewed volume.
    """
    ndi = import_module("scipy", "ndimage", crop)

    (nz, ny, nx) = crop.shape
    dtype = crop.dtype

    nz_new, ny_new, nx_new = (
        len(range(0, nx, resample_factor)),
        ny,
        nx + nz * resample_factor,
    )
    data_reassign = np.zeros((nz_new, ny_new, nx_new), dtype=dtype, like=crop)

    for x in range(nx):
        x_start = x
        x_end = nz * resample_factor + x
        data_reassign[x // resample_factor, :, x_start:x_end:resample_factor] = crop[
            :, :, x
        ].T
    del crop

    # rescale the data, interpolate along z
    data_rescale = ndi.zoom(data_reassign, zoom=(resample_factor, 1, 1), order=1)
    del data_reassign

    data_interp = np.zeros((nz_new, ny_new, nx_new), dtype=dtype, like=data_rescale)

    for z in range(nz_new):
        for k in range(resample_factor):
            data_interp[z, :, k::resample_factor] = data_rescale[
                z * resample_factor - k, :, k::resample_factor
            ]
    del data_rescale

    # rescale the data, to have voxel the same along x an y;
    # remove the first z slice which has artifacts due to resampling
    image_final = ndi.zoom(data_interp[1:], zoom=(1, 1, lateral_scaling), order=1)

    return image_final


def deskew(
    raw_data: ArrayLike,
    pixel_size: float,
    scan_step: float,
    tilt_angle: float,
    num_splits: int = 4,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
) -> np.ndarray:
    """
    Deskew a volume with an angle between the x and z axis.
    Using Bin Yang deskew.
    The deskewing is done per chunks in y-axis.
    Reference: https://github.com/royerlab/dexp/blob/master/dexp/processing/deskew/yang_deskew.py

    Parameters
    ----------
    raw_data : ArrayLike
        The volume to be deskewed
    pixel_size : float
        The pixel size (dx resolution)
    scan_step : float
        The acquisition scan step size
    tilt_angle : float
        The angle in degrees to be deskewed.
    num_splits : int, optional
        Number of splits in y-axis
    to_device : Callable[[ArrayLike], ArrayLike], optional
        Function to move data to device

    Returns
    -------
    np.ndarray
        The deskewed volume.
    """
    resample_factor, lateral_scaling = _scanning_and_scale(
        pixel_size, scan_step, tilt_angle
    )

    deskewed_crops = [
        to_numpy(
            _deskew(
                to_device(crop),
                resample_factor,
                lateral_scaling,
            )
        )
        for crop in np.array_split(raw_data, num_splits, axis=1)
    ]

    return np.concatenate(deskewed_crops, axis=1)
