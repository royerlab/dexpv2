import logging

import pytest
from skimage.data import cells3d
from test_utils import translated_views

from dexpv2.crosscorr import phase_cross_corr
from dexpv2.utils import to_cpu

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy and scipy.")


@pytest.mark.parametrize("perturb_along_axis", [None, 1, 2])
def test_crosscorr(perturb_along_axis: bool, interactive_test: bool) -> None:
    nuclei = xp.asarray(cells3d()[:, 1])

    translation = xp.asarray([5, -5, 10])
    view1, view2 = translated_views(
        nuclei, translation, perturb_along_axis, x_range=7.5, baseline=0.3
    )
    estimated_translation = phase_cross_corr(view1, view2, maximum_shift=0.5)

    if interactive_test:
        import napari

        viewer = napari.Viewer()

        viewer.add_image(to_cpu(view1), blending="additive", colormap="green")
        viewer.add_image(to_cpu(view2), blending="additive", colormap="red")

        napari.run()

    assert xp.allclose(translation, estimated_translation)
