import os
from pathlib import Path

import cupy as cp
import napari
import numpy as np
import tifffile

from dexpv2.cuda import to_numpy, unified_memory
from dexpv2.fusion import dualview_fuse
from dexpv2.registration import estimate_affine_transform


def _downsample(arr: np.ndarray) -> np.ndarray:
    return arr[::4, ::4, ::4]


def main() -> None:
    data_dir = Path(os.environ["BACKTEST_DIR"]) / "daxi_affine"

    v0 = tifffile.imread(data_dir / "v0.tif")[0].astype(np.float32)
    v1 = tifffile.imread(data_dir / "v1.tif")[0].astype(np.float32)

    scale = np.asarray((0.8768124086713188, 0.439, 0.439))
    reg_px_size = 2

    v0 = np.clip(v0, 100, None) - 100
    v1 = np.clip(v1, 100, None) - 100

    _, reg = estimate_affine_transform(
        v0, v1, scale, reg_px_size, return_reg_moving=True
    )

    with unified_memory():
        fused = to_numpy(
            dualview_fuse(cp.asarray(v0), cp.asarray(reg), scale, blending_sigma=5.0)
        )

    viewer = napari.Viewer()

    viewer.add_image(
        _downsample(v0), name="v0", blending="additive", colormap="red", scale=scale
    )
    viewer.add_image(
        _downsample(v1),
        name="v1",
        blending="additive",
        colormap="green",
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        _downsample(reg),
        name="reg",
        blending="additive",
        colormap="blue",
        scale=scale,
    )

    viewer.add_image(
        _downsample(fused),
        name="fused",
        blending="additive",
        colormap="magma",
        scale=scale,
        visible=False,
    )

    napari.run()


if __name__ == "__main__":
    main()
