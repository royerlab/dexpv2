import logging
from typing import Tuple, List

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.constants import DEXPV2_DEBUG
from dexpv2.cuda import import_module
from dexpv2.utils import to_cpu

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

try:
    import cupy as xp

    LOG.info("cupy found.")
except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy.")


def discretize_multiple_f16_to_u16(
    f16_arrays: List[ArrayLike],
) -> Tuple[List[ArrayLike], ArrayLike]:
    """
    Discretizes multiple arrays (e.g., CuPy or NumPy) of float16 values to
    uint16, preserving order using a global mapping across all arrays.

    Parameters
    ----------
    f16_arrays : List[ArrayLike]
        List of input arrays to be discretized. Each array must have dtype
        float16.

    Returns
    -------
        Tuple[List[ArrayLike], ArrayLike]: A tuple containing:
            - u16_list (List[ArrayLike]): A list of discretized arrays,
                                          each with dtype uint16,
                                          corresponding to the input
                                          arrays.
            - u16_to_f16_lut (ArrayLike): A single lookup table (array of
                                          float16) for all arrays, where the
                                          index is the uint16 value and the
                                          value is the corresponding original
                                          float16 value.

    Raises
    ------
        TypeError: If any input array's dtype is not float16.
        ValueError: If the list of arrays is empty, or if the total number
                    of unique values across all arrays exceeds the capacity
                    of uint16 (65536).
    """
    if not f16_arrays:
        raise ValueError("Input list of arrays cannot be empty.")

    # Validate input types and collect original shapes and sizes
    original_shapes = []
    for i, arr in enumerate(f16_arrays):
        if arr.dtype != xp.float16:
            raise TypeError(
                f"Array at index {i} must be an 'xp.ndarray' with dtype xp.float16. "
                f"Got type {type(arr)} with dtype {getattr(arr, 'dtype', 'N/A')}."
            )
        if arr.size == 0:
            raise ValueError(
                f"Array at index {i} is empty. Cannot discretize empty arrays."
            )
        original_shapes.append(arr.shape)

    # Collect unique values and their indices
    uniques, inverses = [], []
    for arr in f16_arrays:
        unq, inv = np.unique(arr, return_inverse=True)
        uniques.append(unq)
        inverses.append(inv.astype(np.uint16))

    # Concatenate all unique values and sort them
    u16_to_f16_lut = np.sort(np.concatenate(uniques))
    if len(u16_to_f16_lut) > np.iinfo(np.uint16).max:
        raise ValueError(
            "The total number of unique values across all arrays exceeds "
            "the capacity of uint16 (65536)."
        )

    # Fix inverses to preserve order
    for unq_k, inv_k in zip(uniques, inverses):
        new_idx_k = np.searchsorted(u16_to_f16_lut, unq_k)
        inv_k[:] = new_idx_k[inv_k]

    # Reshape inverses to match original shapes
    u16_list: list[ArrayLike] = []
    for shape, inv in zip(original_shapes, inverses):
        inv_reshaped = inv.reshape(shape)
        u16_list.append(inv_reshaped)

    return u16_list, u16_to_f16_lut


def reconstruction_by_dilation(
    seed: ArrayLike, mask: ArrayLike, iterations: int
) -> ArrayLike:
    """
    Morphological reconstruction by dilation.
    This function does not compute the full reconstruction.
    The reconstruction is garanteed to be computed fully if the
    number of iterations is equal to the number of pixels.
    See: scikit-image docs for details.

    Parameters
    ----------
    seed : ArrayLike
        Dilation seeds.
    mask : ArrayLike
        Guidance mask.
    iterations : int
        Number of dilations.

    Returns
    -------
        Image reconstructed by dilation.
    """
    ndi = import_module("scipy", "ndimage", seed)

    import numpy as np

    cupy_used = np != xp and not isinstance(seed, np.ndarray)

    lut = None
    # quick-fix for the issue https://github.com/cupy/cupy/issues/9122
    if cupy_used and seed.dtype == xp.float16:
        (seed, mask), lut = discretize_multiple_f16_to_u16([seed, mask])

    seed = np.minimum(seed, mask, out=seed)

    for _ in range(iterations):
        seed = ndi.grey_dilation(seed, size=3, output=seed, mode="constant")
        seed = np.minimum(seed, mask, out=seed)

    if lut is not None:
        # convert back to float16
        seed = xp.take(lut, seed)

    return seed


def fancy_otsu_threshold(
    image: ArrayLike,
    remove_hist_mode: bool = False,
    min_foreground: float = 0.0,
    max_foreground: float = None,
) -> float:
    """
    Compute Otsu threshold with some additional features.
    - Removes histogram mode before computing threshold.
    - Lower bounds the threshold value.

    Parameters
    ----------
    image : ArrayLike
        Input image, IMPORTANT it will be modified in place.
    remove_hist_mode : bool, optional
        Removes histogram mode before computing otsu threshold, useful when background regions are being detected.
    min_foreground : float, optional
        Minimum threshold value, by default 0.0
    max_foreground: float, optional
        Maximum threshold value, by default max value of image

    Returns
    -------
    float
        Threshold value.
    """
    filters = import_module("skimage", "filters", arr=image)
    exposure = import_module("skimage", "exposure", arr=image)

    image = np.sqrt(image)  # deskew data distribution towards left

    # begin thresholding
    robust_max = np.quantile(image, 1 - 1e-6)
    image = np.minimum(robust_max, image, out=image)

    # number of bins according to maximum value
    nbins = int(robust_max / 10)  # binning with window of 10
    nbins = min(nbins, 256)
    nbins = max(nbins, 10)

    LOG.info(f"Estimated almost max. {np.square(robust_max)}")
    LOG.info(f"Histogram with {nbins}")

    hist, bin_centers = exposure.histogram(image, nbins)
    print(len(bin_centers))
    # clip bins and histogram beyond max_foreground value
    if max_foreground is not None:
        below_threshold_mask = bin_centers < np.sqrt(max_foreground)
        bin_centers = bin_centers[below_threshold_mask]
        print(bin_centers)
        hist = hist[below_threshold_mask]

    # histogram disconsidering pixels we are sure are background
    if remove_hist_mode:
        remaining_background_idx = hist.argmax() + 1
        hist = hist[remaining_background_idx:]
        bin_centers = bin_centers[remaining_background_idx:]

    del image
    threshold = np.square(filters.threshold_otsu(hist=(hist, bin_centers)))
    LOG.info(f"Threshold {threshold}")

    threshold = max(threshold, min_foreground)
    LOG.info(f"Threshold after minimum filtering {threshold}")
    return threshold


def subtract_background(
    image: ArrayLike,
    voxel_size: ArrayLike,
    sigma: float = 15.0,
) -> ArrayLike:
    """
    Subtract background using morphological reconstruction by dilation.

    Parameters
    ----------
    image : ArrayLike
        Input image.
    voxel_size : ArrayLike
        Array of voxel size (z, y, x).
    sigma : float, optional
        Sigma used to estimate background, it will be divided by voxel size, by default 15.0
        Lower sigma will remove more background.

    Returns
    -------
    ArrayLike
        Image with background subtracted.
    """
    ndi = import_module("scipy", "ndimage", arr=image)

    sigmas = [sigma / s for s in voxel_size]

    LOG.info(f"Detecting foreground with voxel size {voxel_size} and sigma {sigma}")
    LOG.info(f"Sigmas after scaling {sigmas}")

    seed = ndi.gaussian_filter(image, sigma=sigmas)
    background = reconstruction_by_dilation(seed, image, 100)
    del seed

    foreground = image - background
    del background

    return foreground


def detect_foreground(
    image: ArrayLike,
    voxel_size: ArrayLike,
    sigma: float = 15.0,
    remove_hist_mode: bool = False,
    min_foreground: float = 0.0,
    max_foreground: float = None,
) -> ArrayLike:
    """
    Detect foreground using morphological reconstruction by dilation and thresholding.

    Parameters
    ----------
    image : ArrayLike
        Input image.
    voxel_size : ArrayLike
        Array of voxel size (z, y, x).
    sigma : float, optional
        Sigma used to estimate background, it will be divided by voxel size, by default 15.0
    remove_hist_mode : bool, optional
        Removes histogram mode before computing otsu threshold, useful when background regions are being detected.
    min_foreground : float, optional
        Minimum value of foreground pixels after background subtraction and smoothing, by default 0.0

    Returns
    -------
    ArrayLike
        Binary foreground mask.
    """
    ndi = import_module("scipy", "ndimage")

    foreground = subtract_background(image, voxel_size, sigma=sigma)

    # threshold in smaller image to save memory and sqrt to deskew data distribution towards left
    small_foreground = ndi.zoom(
        foreground, (0.25,) * foreground.ndim, order=1, mode="nearest"
    )

    threshold = fancy_otsu_threshold(
        small_foreground,
        remove_hist_mode=remove_hist_mode,
        min_foreground=min_foreground,
        max_foreground=max_foreground,
    )

    mask = foreground > threshold
    del foreground

    struct = ndi.generate_binary_structure(mask.ndim, 2)
    mask = ndi.binary_opening(mask, structure=struct, output=mask)
    mask = ndi.binary_closing(mask, structure=struct, output=mask)

    if DEXPV2_DEBUG:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(to_cpu(image))
        viewer.add_labels(to_cpu(mask))
        napari.run()

    return mask
