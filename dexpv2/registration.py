import logging

import ants
import numpy as np
from ants import registration
from numpy.typing import ArrayLike

logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)


def estimate_transformation(
    fixed: np.ndarray,
    moving: np.ndarray,
    voxel_size: ArrayLike,
    reg_px_size: int = 1,
) -> np.ndarray:

    LOG.info("Loading data ...")

    ndim = fixed.ndim

    ants_fixed = ants.from_numpy(fixed.astype(np.float32), spacing=tuple(voxel_size))
    ants_moving = ants.from_numpy(moving.astype(np.float32), spacing=tuple(voxel_size))

    ants_fixed = ants.resample_image(ants_fixed, (reg_px_size,) * ndim)
    ants_moving = ants.resample_image(ants_moving, (reg_px_size,) * ndim)
    LOG.info(f"Resampled shape: {ants_fixed.shape}")

    LOG.info("Starting registration ...")
    result = registration(
        ants_fixed,
        ants_moving,
        type_of_transform="Rigid",
        verbose=True,
    )
    LOG.info(f"Result: {result}")

    transform = ants.read_transform(result["fwdtransforms"][0])
    # center = transform.fixed_parameters
    matrix = transform.parameters.reshape((4, 3))
    LOG.info(f"Transforms: {matrix}")

    LOG.info("Applying transformation ...")
    # ants_registred = transform.apply(ants_moving)
    ants_registred = result["warpedmovout"]

    return ants_registred.numpy()


if __name__ == "__main__":
    from pathlib import Path

    import napari
    import tifffile

    root = Path("/mnt/md0/daxi-debugging")

    v0 = tifffile.imread(root / "v0.tif")[0].astype(np.float32)
    v1 = tifffile.imread(root / "v1.tif")[0].astype(np.float32)

    scale = np.asarray((0.8768124086713188, 0.439, 0.439))
    reg_px_size = 2

    # scale *= 2
    # v0 = v0[::2, ::2, ::2]
    # v1 = v1[::2, ::2, ::2]

    v0 = np.clip(v0, 100, None) - 100
    v1 = np.clip(v1, 100, None) - 100

    reg = estimate_transformation(v0, v1, scale, reg_px_size)

    viewer = napari.Viewer()

    viewer.add_image(v0, name="v0", blending="additive", colormap="red", scale=scale)
    viewer.add_image(
        v1, name="v1", blending="additive", colormap="green", scale=scale, visible=False
    )
    viewer.add_image(
        reg,
        name="reg",
        blending="additive",
        colormap="blue",
        scale=(reg_px_size,) * reg.ndim,
    )

    napari.run()
