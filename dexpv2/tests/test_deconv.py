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

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy and scipy.")


def gaussian_freq(size: int, sigma: float, ndim: int) -> ArrayLike:
    line = xp.arange(-size // 2 + 1, size // 2 + 1)

    assert len(line) == size

    grid = xp.stack(xp.meshgrid(*[line for _ in range(ndim)]), axis=-1)
    kernel = xp.exp((-2 * xp.pi * sigma * sigma) * xp.square(grid).sum(axis=-1))

    return kernel


@pytest.mark.parametrize("accelerated", [False, True])
def test_deconv_2d(accelerated: bool, display_test: bool) -> None:
    img = xp.asarray(color.rgb2gray(astronaut()), dtype=xp.float32)

    iterations = 15 if accelerated else 30
    Fkernel = gaussian_freq(img.shape[0], 0.005, ndim=2)
    Fimg = xp.fft.fftn(img)

    blurred = xp.fft.ifftn(Fimg * xp.fft.ifftshift(Fkernel)).real
    blurred = blurred + (xp.random.poisson(lam=25, size=img.shape) - 10) / 255.0

    kernel = xp.fft.fftshift(xp.fft.ifftn(Fkernel))
    deconv = lucy_richardson(
        blurred,
        kernel,
        iterations,
        accelerated=accelerated,
    )

    difference = img - deconv

    if display_test:
        import napari

        viewer = napari.Viewer()

        viewer.add_image(to_cpu(img), blending="additive", colormap="green", name="img")
        viewer.add_image(
            to_cpu(Fimg.real), blending="additive", colormap="green", name="img"
        )
        viewer.add_image(
            to_cpu(Fkernel), blending="additive", colormap="magma", name="Fkernel"
        )
        viewer.add_image(
            to_cpu(xp.square(kernel)).real,
            blending="additive",
            colormap="magma",
            name="kernel",
        )
        viewer.add_image(
            to_cpu(blurred), blending="additive", colormap="red", name="blurred"
        )
        viewer.add_image(
            to_cpu(deconv), blending="additive", colormap="blue", name="deconv"
        )
        viewer.add_image(
            to_cpu(difference), blending="additive", colormap="magma", name="difference"
        )

        napari.run()

    # TODO define a value of reconstruction error check
    assert xp.mean(xp.abs(difference)) < 1
