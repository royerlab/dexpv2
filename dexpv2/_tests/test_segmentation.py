import logging

from skimage.data import cells3d

from dexpv2.segmentation import detect_foreground
from dexpv2.utils import to_cpu

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy.")


def test_foreground_detection(interactive_test: bool) -> None:
    # TODO: think about a better testing scheme
    nuclei = xp.asarray(cells3d()[:, 1])
    foreground = detect_foreground(nuclei, [1, 1, 1], sigma=50)

    if interactive_test:
        import napari

        viewer = napari.Viewer()

        viewer.add_image(to_cpu(nuclei), blending="additive", colormap="magma")
        viewer.add_labels(to_cpu(foreground))

        napari.run()


def test_reconstruction_by_dilation_modes() -> None:
    import numpy as np
    from scipy import ndimage

    from dexpv2.segmentation import reconstruction_by_dilation

    rng = np.random.default_rng(0)
    mask = (rng.random((16, 32, 32)) ** 3 * 800).astype(np.float32)
    seed = ndimage.gaussian_filter(mask, 2)

    # The default is unchanged: zero padding.
    expected = np.minimum(seed, mask)
    for _ in range(10):
        expected = np.minimum(
            ndimage.grey_dilation(expected, size=3, mode="constant"), mask
        )
    result = reconstruction_by_dilation(xp.asarray(seed), xp.asarray(mask), 10)
    np.testing.assert_array_equal(to_cpu(result), expected)

    # "nearest" matches it on non-negative data, and runs on float16.
    nearest = reconstruction_by_dilation(
        xp.asarray(seed), xp.asarray(mask), 10, mode="nearest"
    )
    np.testing.assert_array_equal(to_cpu(nearest), expected)
    half = reconstruction_by_dilation(
        xp.asarray(seed, dtype=xp.float16),
        xp.asarray(mask, dtype=xp.float16),
        10,
        mode="nearest",
    )
    assert half.dtype == xp.float16
