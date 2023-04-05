import logging

import pytest
from numpy.typing import ArrayLike
from skimage import color
from skimage.data import astronaut

from dexpv2.deconv import lucy_richardson
from dexpv2.utils import to_cpu

LOG = logging.getLogger(__name__)

try:
    import cupy as xp
    import cupyx.scipy.ndimage as ndi

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp
    import scipy.ndimage as ndi

    LOG.info("cupy not found using numpy and scipy.")


def gaussian_kernel(size: int, sigma: float, ndim: int) -> ArrayLike:
    line = xp.arange(-size // 2 + 1, size // 2 + 1)

    assert len(line) == size
    assert line[size // 2] == 0.0

    grid = xp.stack(xp.meshgrid(*[line for _ in range(ndim)]), axis=-1)
    kernel = xp.exp(-xp.square(grid).sum(axis=-1) / (2 * sigma**2))
    kernel = kernel / xp.sum(kernel)

    return kernel


@pytest.mark.parametrize("accelerated", [False, True])
def test_deconv_2d(accelerated: bool, display_test: bool) -> None:
    img = xp.asarray(color.rgb2gray(astronaut()), dtype=xp.float32)

    iterations = 15 if accelerated else 30
    kernel = xp.ones((5,) * img.ndim)

    blurred = (
        ndi.convolve(img, kernel)
        + (xp.random.poisson(lam=25, size=img.shape) - 10) / 255.0
    )

    deconv = lucy_richardson(
        blurred,
        kernel,
        iterations,
        accelerated=accelerated,
    )

    if display_test:
        import napari

        viewer = napari.Viewer()

        viewer.add_image(to_cpu(img), blending="additive", colormap="green")
        viewer.add_image(to_cpu(kernel), blending="additive", colormap="magma")
        viewer.add_image(to_cpu(blurred), blending="additive", colormap="red")
        viewer.add_image(to_cpu(deconv), blending="additive", colormap="blue")

        napari.run()

    # TODO add reconstruction error check


# @pytest.mark.parametrize("accelerated", [False, True])
# def test_deconv_3d(accelerated: bool, display_test: bool) -> None:
#     data = color.rgb2gray(astronaut)
#     membrane = xp.asarray(data[:, 0], dtype=xp.float32)
#     nuclei = xp.asarray(data[:, 1], dtype=xp.float32)
#
#     iterations = 10 if accelerated else 20
#     kernel = gaussian_kernel(5, 1.0, membrane.ndim)
#
#     for img in (nuclei, membrane):
#         blurred = ndi.convolve(img, kernel)
#         deconv = lucy_richardson(
#             blurred,
#             kernel,
#             iterations,
#             accelerated=accelerated,
#         )
#
#         if display_test:
#             import napari
#             viewer = napari.Viewer()
#
#             viewer.add_image(to_cpu(img), blending="additive", colormap="green")
#             viewer.add_image(to_cpu(kernel), blending="additive", colormap="magma")
#             viewer.add_image(to_cpu(blurred), blending="additive", colormap="red")
#             viewer.add_image(to_cpu(deconv), blending="additive", colormap="blue")
#
#             napari.run()
#
