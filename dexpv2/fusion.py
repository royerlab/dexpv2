import numpy as np
from numpy.typing import ArrayLike

from dexpv2.utils import translation_slicing


def fuse_multiview(
    C0L0: ArrayLike,
    C0L1: ArrayLike,
    C1L0: ArrayLike,
    C1L1: ArrayLike,
    camera_1_translation: ArrayLike,
    camera_1_flip: bool,
) -> ArrayLike:
    """
    Fuse views from a multi-view microscope.
    It assumes that the views were captured using two cameras and two light sheets.
    The views from different cameras can be flipped and translated.
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
    camera_1_translation : ArrayLike
        Translation between camera 1 and camera 0 (reference).
    camera_1_flip : ArrayLike
        Indicates if camera 1 is flipped on the last axis.

    Returns
    -------
    ArrayLike
        Fused image.
    """
    camera_0 = (C0L0 + C0L1) / 2
    camera_1 = (C1L0 + C1L1) / 2

    if camera_1_flip:
        camera_1 = np.flip(camera_1, axis=-1)

    ref_slice = translation_slicing(-camera_1_translation)
    mov_slice = translation_slicing(camera_1_translation)

    fused = camera_0
    fused[ref_slice] = (fused[ref_slice] + camera_1[mov_slice]) / 2

    return fused
