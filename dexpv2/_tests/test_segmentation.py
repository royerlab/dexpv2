import logging

from skimage.data import cells3d
import pytest

from dexpv2.segmentation import detect_foreground, reconstruction_by_dilation
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


def test_foreground_detection_with_float16() -> None:
    # Test with float16 dat
    # a
    nuclei = xp.asarray(cells3d()[:, 1])
    nuclei = nuclei / nuclei.max()
    nuclei = nuclei.astype(xp.float16)
    mask = xp.copy(nuclei)

    # Ensure we are using cupy backend
    import numpy as np

    if isinstance(nuclei, np.ndarray):
        pytest.skip("Skipping test as cupy is not available.")

    foreground_cp = reconstruction_by_dilation(nuclei, mask, iterations=10)
    foreground_cp = to_cpu(foreground_cp)

    nuclei_f32 = nuclei.astype(xp.float32)
    mask_f32 = mask.astype(xp.float32)
    foreground_f32 = reconstruction_by_dilation(nuclei_f32, mask_f32, iterations=10)

    # Convert to numpy for comparison
    # Obs. skimage operations won't work with np.float16 so we need to convert
    # to float32 and hope that the conversion doesn't change the result too much
    nuclei_np = to_cpu(nuclei_f32)
    mask_np = to_cpu(mask_f32)
    foreground_np = reconstruction_by_dilation(nuclei_np, mask_np, iterations=10)

    # Check that the output is a binary mask
    assert np.allclose(foreground_cp, foreground_np)
    assert np.allclose(foreground_cp, foreground_f32)
