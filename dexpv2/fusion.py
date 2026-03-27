import logging
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.cuda import import_module
from dexpv2.intensity import equalize_views
from dexpv2.registration import apply_affine_transform
from dexpv2.utils import translation_slicing

LOG = logging.getLogger(__name__)


def multiview_fuse(
    C0L0: ArrayLike,
    C0L1: ArrayLike,
    C1L0: ArrayLike,
    C1L1: ArrayLike,
    camera_1_T: ArrayLike,
    camera_1_flip: bool,
    L1_over_L0_ratio: Optional[float] = None,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
) -> ArrayLike:
    """
    Fuse views from a multi-view microscope.
    It assumes that the views were captured using two cameras and two light sheets.
    The views from different cameras can be flipped and affine-transformed.
    Translations are rounded and done at the pixel resolution.

    Parameters
    ----------
    C0L0 : ArrayLike
        View from camera 0 and light sheet 0.
    C0L1 : ArrayLike
        View from camera 0 and light sheet 0.
    C1L0 : ArrayLike
        View from camera 1 and light sheet 0.
    C1L1 : ArrayLike
        View from camera 1 and light sheet 1.
    camera_1_T : ArrayLike
        Transformation between camera 1 and camera 0 (reference).
        When provided with a 1D array, it is interpreted as an integer translation in pixel space.
        When provided with a 2D array, it is interpreted as an affine transformation in physical space.
    camera_1_flip : ArrayLike
        Indicates if camera 1 is flipped on the last axis.
    L1_over_L0_ratio : float, optional
        Ratio between the light sheet intensity of camera 1 and camera 0.
    to_device : Callable, optional
        Helper function to send data to specialized device, this function sends the data
        to the device only when needed, reducing the memory usage.

    Returns
    -------
    ArrayLike
        Fused image.
    """
    if L1_over_L0_ratio is None:
        camera_0 = (to_device(C0L0).astype(np.float32) + to_device(C0L1)) * 0.5
        camera_1 = (to_device(C1L0).astype(np.float32) + to_device(C1L1)) * 0.5

    elif L1_over_L0_ratio > 1.0:
        camera_0 = (
            to_device(C0L0).astype(np.float32) * L1_over_L0_ratio + to_device(C0L1)
        ) * 0.5
        camera_1 = (
            to_device(C1L0).astype(np.float32) * L1_over_L0_ratio + to_device(C1L1)
        ) * 0.5

    else:
        L0_over_L1_ratio = 1 / L1_over_L0_ratio
        camera_0 = (
            to_device(C0L0) + to_device(C0L1).astype(np.float32) * L0_over_L1_ratio
        ) * 0.5
        camera_1 = (
            to_device(C1L0) + to_device(C1L1).astype(np.float32) * L0_over_L1_ratio
        ) * 0.5

    if camera_0.dtype != np.float32:
        LOG.warning(
            f"fusion array cast to {camera_0.dtype} using more memory than expected."
        )

    if camera_1_flip:
        camera_1 = np.flip(camera_1, axis=-1)

    camera_1_T = np.asarray(camera_1_T, like=camera_0)
    if camera_1_T.ndim > 1:
        camera_1 = apply_affine_transform(camera_1_T, camera_1, voxel_size=None)
        camera_0 = (camera_0 + camera_1) * 0.5
    else:
        ref_slice = translation_slicing(-camera_1_T)
        mov_slice = translation_slicing(camera_1_T)

        camera_0[ref_slice] += camera_1[mov_slice]
        camera_0[ref_slice] /= 2

    return camera_0


def dualview_fuse(
    V0: ArrayLike,
    V1: ArrayLike,
    voxel_size: ArrayLike,
    blending_sigma: float = 0.0,
    equalize: bool = False,
) -> ArrayLike:
    """
    Fuse two views, V0 and V1, to create a combined image.

    This function combines two images by fusing their intensities while considering voxel-wise foreground regions.
    It optionally applies Gaussian blurring to the foreground regions for blending.

    IMPORTANT: Images must be registered before fusion.

    Parameters
    ----------
    V0 : ArrayLike
        The first image to be fused.
    V1 : ArrayLike
        The second image to be fused.
    voxel_size : ArrayLike
        The physical voxel size of the images.
    blending_sigma : float, optional
        The standard deviation for Gaussian blurring of foreground regions. Default is 0.0 (no blending).
    equalize : bool, optional
        When True, the intensity of the views are equalized to the brightest view. Default is False.

    Returns
    -------
    ArrayLike
        The fused image, with intensity values blended based on foreground regions.

    Example
    -------
    >>> fused_image = dualview_fuse(V0, V1, voxel_size=(1, 1, 1), blending_sigma=1.0)
    """

    foreground_v0 = V0 > 0
    foreground_v0 = foreground_v0.astype(np.float32)

    foreground_v1 = V1 > 0
    foreground_v1 = foreground_v1.astype(np.float32)

    if blending_sigma > 0:
        ndi = import_module("scipy", "ndimage")
        sigma = [blending_sigma / v for v in voxel_size]

        ndi.gaussian_filter(foreground_v0, sigma=sigma, output=foreground_v0)
        ndi.gaussian_filter(foreground_v1, sigma=sigma, output=foreground_v1)

    foreground_total = foreground_v0 + foreground_v1
    foreground_total = np.where(foreground_total < 1e-6, 1, foreground_total)

    if equalize:
        V0, V1 = equalize_views(V0, V1)

    foreground_v0 *= V0
    foreground_v1 *= V1
    foreground_v0 += foreground_v1
    foreground_v0 /= foreground_total

    return foreground_v0
