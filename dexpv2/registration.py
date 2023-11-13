import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import ants
import numpy as np
from ants import registration
from numpy.typing import ArrayLike

from dexpv2.cuda import import_module

LOG = logging.getLogger(__name__)


def estimate_affine_transform(
    fixed: np.ndarray,
    moving: np.ndarray,
    voxel_size: ArrayLike,
    reg_px_size: Optional[int] = None,
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

    if reg_px_size is None:
        resampled_ants_fixed = ants_fixed
        resampled_ants_moving = ants_moving
    else:
        resampled_ants_fixed = ants.resample_image(ants_fixed, (reg_px_size,) * ndim)
        resampled_ants_moving = ants.resample_image(ants_moving, (reg_px_size,) * ndim)

    LOG.info(f"Resampled shape: {resampled_ants_fixed.shape}")

    LOG.info("Starting registration ...")
    result = registration(
        resampled_ants_fixed,
        resampled_ants_moving,
        type_of_transform="Affine",
        verbose=verbose,
        **kwargs,
    )
    LOG.info(f"Result: {result}")

    if len(result["fwdtransforms"]) != 1:
        raise ValueError(
            f"Expected only one transform. Found {len(result['fwdtransforms'])}"
        )

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
    voxel_size: Optional[ArrayLike],
) -> np.ndarray:
    """
    Apply an affine transformation to a 3D image.

    Modified from Ed's mantin's code.

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
    ndi = import_module("scipy", "ndimage")

    A = transform.parameters.reshape((3, 4), order="F")
    A[:, :3] = A[:, :3].transpose()

    # Reference:
    # https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
    # https://github.com/netstim/leaddbs/blob/a2bb3e663cf7fceb2067ac887866124be54aca7d/helpers/ea_antsmat2mat.m
    # T = original translation offset from A
    # T = T + (I - A) @ centering
    A[:, -1] += (np.eye(3) - A[:3, :3]) @ transform.fixed_parameters

    if voxel_size is not None:
        # transformations are dome on physical space, therefore we map to physical space and then back.
        inv_scaling = np.diag(1 / np.asarray(voxel_size))
        scaling = np.eye(4)
        scaling[:3, :3] = np.diag(voxel_size)
        A = inv_scaling @ A @ scaling

    A = np.asarray(A, like=image)

    transformed = ndi.affine_transform(image, A, order=1)

    return transformed
