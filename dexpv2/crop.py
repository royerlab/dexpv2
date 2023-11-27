from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import find_objects

from dexpv2.cuda import import_module
from dexpv2.utils import to_cpu


def foreground_bbox(
    image: ArrayLike,
    voxel_size: ArrayLike,
    downscale: int = 16,
) -> ArrayLike:
    """
    Compute bounding box of the largest foreground object.

    Parameters
    ----------
    image : ArrayLike
        N-dimensional grayscale image.
    voxel_size : ArrayLike
        Voxel size in each dimension.
    downscale : int
        Downscaling factor to speed-up computation.

    Returns
    -------
    ArrayLike
        Bounding box 2N array of (0_start, 1_start, ..., 0_end, 1_end, ...).
    """
    ndi = import_module("scipy", "ndimage")
    morph = import_module("skimage", "morphology")

    scaling = np.asarray([v / downscale for v in voxel_size])

    image = ndi.zoom(image, scaling, order=1)

    foreground = image > np.mean(image)

    if image.ndim == 2:
        struct = morph.disk(3)
    elif image.ndim == 3:
        struct = morph.ball(3)
    else:
        raise ValueError(
            f"Unsupported image dimension {image.ndim}. Only 2D and 3D allowed."
        )

    foreground = morph.binary_closing(foreground, footprint=struct)

    labels, n_labels = ndi.label(foreground)

    area = ndi.sum_labels(
        foreground, labels, index=np.arange(1, n_labels + 1, like=foreground)
    )

    largest = np.argmax(area) + 1

    obj = find_objects(to_cpu(labels == largest))[0]

    bbox = np.asarray([[s.start for s in obj], [s.stop for s in obj]])

    bbox = np.round(bbox / scaling).astype(int)

    return bbox.reshape(-1)


def find_moving_bboxes_consensus(
    bboxes: ArrayLike,
    shifts: ArrayLike,
    quantile: float = 0.9,
) -> ArrayLike:
    """
    Find consensus bounding box of moving objects.

    Parameters
    ----------
    bboxes : ArrayLike
        Bounding boxes of moving objects (N, 2D) array.
    shifts : ArrayLike
        Shifts of moving objects (N, D) array.
    quantile : float
        Quantile to use for consensus bounding box size.

    Returns
    -------
    ArrayLike
        Fixed bounding boxes, (N, 2D) array.
    """
    bboxes = np.asarray(bboxes)
    shifts = np.asarray(shifts)

    if bboxes.shape[0] != shifts.shape[0]:
        raise ValueError(
            f"Number of bboxes ({bboxes.shape[0]}) does not match number of shifts ({shifts.shape[0]})."
        )

    ndim = shifts.shape[1]
    if bboxes.shape[1] != 2 * ndim:
        raise ValueError(
            f"Number of bbox dimensions ({bboxes.shape[1]}) does not match number of shift dimensions ({ndim} x 2)."
        )

    cum_shifts = np.cumsum(shifts, axis=0)

    start = np.quantile(bboxes[:, :ndim], 1 - quantile, axis=0)[None, ...] + cum_shifts
    start = np.clip(start, 0, None)

    size = np.quantile(bboxes[:, ndim:] - bboxes[:, :ndim], quantile, axis=0)
    end = start + size

    return np.round(np.concatenate([start, end], axis=1)).astype(int)


def to_slice(bbox: ArrayLike) -> Tuple[slice, ...]:
    """
    Convert bounding box to slice.

    Parameters
    ----------
    bbox : ArrayLike
        Bounding box 2N array of (0_start, 1_start, ..., 0_end, 1_end, ...).

    Returns
    -------
    tuple(slice)
        Slice object.
    """
    bbox = np.asarray(bbox)
    ndim = bbox.size // 2
    return tuple(slice(start, end) for start, end in zip(bbox[:ndim], bbox[ndim:]))
