import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import ants
import numpy as np
from ants import registration
from numpy.typing import ArrayLike

logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)


def estimate_affine_transform(
    fixed: np.ndarray,
    moving: np.ndarray,
    voxel_size: ArrayLike,
    reg_px_size: int,
    return_reg_moving: bool = False,
    output_path: Optional[Path] = None,
    verbose: bool = False,
    **kwargs,
) -> Union[ants.ANTsTransform, Tuple[ants.ANTsTransform, np.ndarray]]:
    """
    Estimate an affine transformation between two 3D images.

    Parameters
    ----------
    fixed : np.ndarray
        The fixed image as a NumPy array.
    moving : np.ndarray
        The moving image as a NumPy array.
    voxel_size : ArrayLike
        The physical voxel size of the images.
    reg_px_size : int
        The physical size of the resampled images for registration.
    return_reg_moving : bool, optional
        If True, return the transformed moving image.
    output_path : Optional[Path], optional
        The path to save the estimated transformation.
    verbose : bool, optional
        If True, display verbose registration information.
    **kwargs
        Additional keyword arguments passed to the ANTs registration function.

    Returns
    -------
    ants.ANTsTransform
        The estimated affine transformation.
    Tuple[ants.ANTsTransform, np.ndarray]
        If return_reg_moving is True, also returns the transformed moving image.

    Example
    -------
    >>> transform = estimate_affine_transform(fixed_img, moving_img, voxel_size=(1, 1, 1), reg_px_size=64)
    """
    LOG.info("Loading data ...")

    ndim = fixed.ndim

    ants_fixed = ants.from_numpy(fixed.astype(np.float32), spacing=tuple(voxel_size))
    ants_moving = ants.from_numpy(moving.astype(np.float32), spacing=tuple(voxel_size))

    small_ants_fixed = ants.resample_image(ants_fixed, (reg_px_size,) * ndim)
    small_ants_moving = ants.resample_image(ants_moving, (reg_px_size,) * ndim)
    LOG.info(f"Resampled shape: {small_ants_fixed.shape}")

    LOG.info("Starting registration ...")
    result = registration(
        small_ants_fixed,
        small_ants_moving,
        type_of_transform="Rigid",
        verbose=verbose,
        **kwargs,
    )
    LOG.info(f"Result: {result}")

    transform = ants.read_transform(result["fwdtransforms"][0])

    if output_path is not None:
        LOG.info(f"Saving result to {output_path} ...")
        # copying
        ants.write_transform(transform, str(output_path))

    matrix = transform.parameters.reshape((4, 3))
    LOG.info(f"Transform: {matrix}")

    if return_reg_moving:
        reg_moving = transform.apply_to_image(ants_moving)
        return transform, reg_moving.numpy().astype(moving.dtype)

    return transform


def apply_affine_transform(
    transform: ants.ANTsTransform,
    image: np.ndarray,
    voxel_size: ArrayLike,
) -> np.ndarray:
    """
    Apply an affine transformation to a 3D image.

    Parameters
    ----------
    transform : ants.ANTsTransform
        The affine transformation to apply.
    image : np.ndarray
        The input image as a NumPy array.
    voxel_size : ArrayLike
        The physical voxel size of the image.

    Returns
    -------
    np.ndarray
        The transformed image as a NumPy array.

    Example
    -------
    >>> transformed_img = apply_affine_transform(affine_transform, input_img, voxel_size=(1, 1, 1))
    """
    ants_image = ants.from_numpy(image.astype(np.float32), spacing=tuple(voxel_size))
    transformed = transform.apply_to_image(ants_image)
    return transformed.numpy().astype(image.dtype)
