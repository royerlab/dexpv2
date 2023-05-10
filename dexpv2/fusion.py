import logging
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.utils import translation_slicing

LOG = logging.getLogger(__name__)


def multiview_fuse(
    C0L0: ArrayLike,
    C0L1: ArrayLike,
    C1L0: ArrayLike,
    C1L1: ArrayLike,
    camera_1_translation: ArrayLike,
    camera_1_flip: bool,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
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
    to_device : Callable, optional
        Helper function to send data to specialized device, this function sends the data
        to the device only when needed, reducing the memory usage.

    Returns
    -------
    ArrayLike
        Fused image.
    """
    camera_0 = (to_device(C0L0).astype(np.float32) + to_device(C0L1)) / 2
    camera_1 = (to_device(C1L0).astype(np.float32) + to_device(C1L1)) / 2

    if camera_0.dtype != np.float32:
        LOG.warning(
            f"fusion array cast to {camera_0.dtype} using more memory than expected."
        )

    camera_1_translation = np.asarray(camera_1_translation, like=camera_0)
    ref_slice = translation_slicing(-camera_1_translation)
    mov_slice = translation_slicing(camera_1_translation)

    if camera_1_flip:
        camera_1 = np.flip(camera_1, axis=-1)

    camera_0[ref_slice] += camera_1[mov_slice]
    camera_0[ref_slice] /= 2

    return camera_0
