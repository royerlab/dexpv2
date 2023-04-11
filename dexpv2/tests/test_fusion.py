import logging

from skimage.data import cells3d
from test_utils import translated_views

from dexpv2.fusion import fuse_multiview
from dexpv2.utils import normalize, to_cpu, translation_slicing

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy and scipy.")


def test_fusion(display_test: bool) -> None:
    nuclei = xp.asarray(cells3d()[:, 1])

    translation = xp.asarray([5, -5, 10])
    baseline_blending = 0.6
    blending_kwargs = {"x_range": 7.5, "baseline": baseline_blending}

    camera_0, camera_1 = translated_views(
        nuclei,
        translation,
        -1,
        **blending_kwargs,
    )
    camera_1 = xp.flip(camera_1, axis=-1)

    zeros = xp.zeros(nuclei.ndim, dtype=int)
    C0L0, C0L1 = translated_views(camera_0, zeros, 1, **blending_kwargs)
    C1L0, C1L1 = translated_views(camera_1, zeros, 1, **blending_kwargs)

    fused = fuse_multiview(C0L0, C0L1, C1L0, C1L1, translation, camera_1_flip=True)

    crop_nuclei = nuclei[translation_slicing(translation)]

    difference = normalize(fused) - normalize(crop_nuclei)
    difference = difference[
        translation_slicing(-translation)
    ]  # removing region without overlap

    if display_test:
        import napari

        viewer = napari.Viewer()
        kwargs = {
            "blending": "additive",
            "colormap": "green",
            "contrast_limits": (0, nuclei.max().item()),
        }

        viewer.add_image(
            to_cpu(difference), blending="additive", colormap="magma", name="difference"
        )
        viewer.add_image(
            to_cpu(crop_nuclei), blending="additive", colormap="red", name="nuclei"
        )
        viewer.add_image(
            to_cpu(fused), blending="additive", colormap="blue", name="fused"
        )
        viewer.add_image(to_cpu(C0L0), name="C0L0", **kwargs)
        viewer.add_image(to_cpu(C0L1), name="C0L1", **kwargs)
        viewer.add_image(to_cpu(C1L0), name="C1L0", **kwargs)
        viewer.add_image(to_cpu(C1L1), name="C1L1", **kwargs)

        viewer.grid.shape = (-1, 2)
        viewer.grid.enabled = True

        napari.run()

    assert xp.abs(difference).mean() < 0.05
