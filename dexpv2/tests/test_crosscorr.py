import logging
from typing import Tuple

import pytest
from numpy.typing import ArrayLike
from skimage.data import cells3d

from dexpv2.crosscorr import phase_cross_corr
from dexpv2.utils import to_cpu

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy and scipy.")


def _generate_blending(length: int) -> ArrayLike:
    """Creates a blending map using a sigmoid function given a range and a minimum baseline value."""
    value_range = 7.5
    minimum_value = 0.3
    x = xp.linspace(-value_range, value_range, length)
    y = 1 / (1 + xp.exp(-x))
    y = (y + minimum_value) / (1 + minimum_value)
    return y


def _get_slice(value: int) -> slice:
    """Helper function to get correct slicing from negative or positive integers."""
    if value < 0:
        return slice(None, value)
    return slice(value, None)


def _translated_views(
    img: ArrayLike,
    translation: ArrayLike,
    perturb_half: bool = False,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Generates translated views of the input image.
    Additional perturbation of opposite halves are possible to simulate a lightsheet dataset.

    Parameters
    ----------
    img : ArrayLike
        Input image.
    translation : ArrayLike
        Translation between views.
    perturb_half : bool, optional
        When true opposite halfs of the image are perturbed, by default False

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Tuple of two views of the original image displaced by `translation`.
    """

    img1 = img[tuple(_get_slice(t) for t in translation)]
    img2 = img[tuple(_get_slice(-t) for t in translation)]

    assert img1.shape == img2.shape

    if perturb_half:
        blending = _generate_blending(img1.shape[-1])
        img1 = img1 * blending[None, None, ...]
        img2 = img2 * xp.flip(blending)[None, None, ...]

    return img1, img2


@pytest.mark.parametrize("perturb_half", [False, True])
def test_crosscorr(perturb_half: bool, display_test: bool) -> None:
    nuclei = xp.asarray(cells3d()[:, 1])

    translation = [5, -5, 10]
    view1, view2 = _translated_views(nuclei, translation, perturb_half)
    estimated_translation = phase_cross_corr(view1, view2, maximum_shift=0.5)

    if display_test:
        import napari

        viewer = napari.Viewer()

        viewer.add_image(to_cpu(view1), blending="additive", colormap="green")
        viewer.add_image(to_cpu(view2), blending="additive", colormap="red")

        napari.run()

    assert xp.allclose(translation, estimated_translation)
