import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import find_objects

from dexpv2.cuda import import_module, to_numpy

LOG = logging.getLogger(__name__)


def foreground_bbox(
    image: ArrayLike,
    voxel_size: ArrayLike,
    downscale: int = 20,
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

    original_shape = np.asarray(image.shape)
    scaling = np.asarray([v / downscale for v in voxel_size])

    image = ndi.zoom(image, scaling, order=1, mode="nearest")

    foreground = image > np.mean(image)

    if image.ndim == 2:
        struct = morph.disk(5)
    elif image.ndim == 3:
        struct = morph.ball(5)
    else:
        raise ValueError(
            f"Unsupported image dimension {image.ndim}. Only 2D and 3D allowed."
        )

    foreground = morph.binary_closing(foreground, footprint=struct)

    labels, _ = ndi.label(foreground)

    start = np.asarray(labels.shape)
    end = np.zeros_like(start)

    # 320 ^ 3 pixels in physical units
    # minimum size to be a meaningful object
    min_size = np.prod(scaling * 320)
    LOG.warning("Min size: %s", min_size)

    for obj in find_objects(to_numpy(labels)):
        if obj is None:
            continue
        area = np.prod([s.stop - s.start for s in obj])
        LOG.warning("Area: %s, obj slicing: %s", area, obj)
        if area < min_size:
            continue
        LOG.warning("Passed area check.")
        start = np.minimum(start, [s.start for s in obj])
        end = np.maximum(end, [s.stop for s in obj])

    if np.any(start > end):
        LOG.warning("No foreground object found.")
        bbox = np.concatenate([np.zeros_like(start), original_shape])
    else:
        start = np.maximum(start - 1, 0) / scaling
        end = (end + 2) / scaling  # 1 padding plus 1 to compensate for start
        bbox = np.concatenate([start, end])
        bbox = np.round(bbox).astype(int)

    return bbox


def find_moving_bboxes_consensus(
    bboxes: ArrayLike,
    shifts: ArrayLike,
    shape: ArrayLike,
    quantile: float = 0.99,
    outlier_threshold: Optional[float] = 2,
) -> ArrayLike:
    """
    Find consensus bounding box of moving objects.
    First 2 * D dimensions are start and end of source bounding boxes.
    Last 2 * D dimensions are start and end of destination bounding boxes.

    Parameters
    ----------
    bboxes : ArrayLike
        Bounding boxes of moving objects (N, 2 * D) array.
    shifts : ArrayLike
        Shifts of moving objects (N, D) array.
    shape : ArrayLike
        Volume shape.
    quantile : float
        Quantile to use for consensus bounding box size.
    outlier_threshold : float
        Threshold for outlier shifts, multiples of standard deviation of shifts.

    Returns
    -------
    ArrayLike
        Fixed bounding boxes, (N, 4 * D) array.
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

    if outlier_threshold is not None:
        shift_std = shifts.std(axis=0)
        mean_shift = shifts.mean(axis=0)
        outlier_shift = np.abs(shifts - mean_shift) > outlier_threshold * shift_std
        shifts[outlier_shift] = 0

        print(f"Mean shift: {mean_shift}")
        print(f"Shift std: {shift_std}")
        print(f"Outlier shifts found: {outlier_shift.sum()}")

    # accumulate shifts
    cum_shifts = np.cumsum(shifts, axis=0)

    # enlarged bounding boxes
    bboxes_size = bboxes[:, ndim:] - bboxes[:, :ndim]
    suggested_size = np.quantile(bboxes_size, quantile, axis=0)
    suggested_size = np.round(suggested_size).astype(int)

    diff_from_size = np.clip(suggested_size - bboxes_size, 0, None)

    src_start = bboxes[:, :ndim] - diff_from_size // 2
    src_start = np.clip(src_start, 0, None)
    src_start = np.round(src_start).astype(int)

    src_end = bboxes[:, ndim:] + diff_from_size // 2
    src_end = np.minimum(src_end, shape)
    src_end = np.round(src_end).astype(int)

    # apply shift to destination bounding box
    dst_start = src_start - cum_shifts
    dst_start = dst_start - dst_start.min(axis=0)
    dst_end = dst_start + src_end - src_start

    return np.round(
        np.concatenate([src_start, src_end, dst_start, dst_end], axis=1)
    ).astype(int)


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


def fix_crop_slice(
    crop_slice: Tuple[slice, ...],
    source_shape: Tuple[int, ...],
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """
    Fix crop slice when it exceeds the source shape.

    Parameters
    ----------
    crop_slice : Tuple[slice, ...]
        Cropping slicing tuple, must have .start and .stop.
    source_shape : Tuple[int, ...]
        Source (cropped object) shape.

    Returns
    -------
    Tuple[Tuple[slice, ...], Tuple[slice, ...]]
        Fixed source slice and destination slice to match cropping.
    """
    fixed_slice = tuple(
        slice(max(0, slc.start), min(src, slc.stop))
        for slc, src in zip(crop_slice, source_shape)
    )

    dst_slice = tuple(slice(0, f.stop - f.start) for f in fixed_slice)

    return fixed_slice, dst_slice
