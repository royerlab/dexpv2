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
