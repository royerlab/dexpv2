import logging

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.constants import DEXPV2_DEBUG
from dexpv2.cuda import import_module
from dexpv2.utils import to_cpu

LOG = logging.getLogger(__name__)


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
    ndi = import_module("scipy", "ndimage")

    seed = np.minimum(seed, mask)  # just making sure

    for _ in range(iterations):
        seed = ndi.grey_dilation(seed, size=3, output=seed, mode="constant")
        seed = np.minimum(seed, mask)

    return seed


def detect_foreground(
    image: ArrayLike,
    voxel_size: ArrayLike,
    sigma: float = 15.0,
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

    Returns
    -------
    ArrayLike
        Binary foreground mask.
    """
    ndi = import_module("scipy", "ndimage")
    filters = import_module("skimage", "filters")

    sigmas = [sigma / s for s in voxel_size]

    LOG.info(f"Detecting foreground with voxel size {voxel_size} and sigma {sigma}")
    LOG.info(f"Sigmas after scaling {sigmas}")

    seed = ndi.gaussian_filter(image, sigma=sigmas)
    background = reconstruction_by_dilation(seed, image, 100)
    del seed

    foreground = image - background
    del background

    # threshold in smaller image to save memory and sqrt to deskew data distribution towards left
    small_foreground = np.sqrt(ndi.zoom(foreground, (0.25,) * foreground.ndim, order=1))
    mask = foreground > np.square(filters.threshold_otsu(small_foreground))
    del foreground, small_foreground

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
