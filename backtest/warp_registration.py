import os
from pathlib import Path
from typing import Optional, Tuple

import cupy as cp
import napari
import numpy as np
import tifffile

from dexpv2.cuda import to_numpy, unified_memory
from dexpv2.warp import apply_warp, estimate_multiscale_warp


def _downsample(
    arr: np.ndarray, step: int, axis: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    if axis is None:
        axis = tuple(range(arr.ndim))

    slicing = [slice(None)] * arr.ndim
    for i in axis:
        slicing[i] = slice(None, None, step)

    return arr[tuple(slicing)]


def main() -> None:
    data_dir = Path(os.environ["BACKTEST_DIR"]) / "daxi_warp_registration"

    scale = [0.8768124086713188, 0.439, 0.439]

    v0 = tifffile.imread(data_dir / "v0.tif")
    v1 = tifffile.imread(data_dir / "v1.tif")

    v0 = _downsample(v0, 2)
    v1 = _downsample(v1, 2)

    with unified_memory():
        warp_field = estimate_multiscale_warp(
            v1,
            v0,
            n_scales=6,
            tile=(8, 32, 32),
            overlap=(4, 16, 16),
            score_threshold=0.5,
            to_device=cp.asarray,
        )

        print("Warp field score range", warp_field[-1].min(), warp_field[-1].max())

        warped_v0 = to_numpy(apply_warp(cp.asarray(v0, cp.float32), warp_field))

    viewer = napari.Viewer()

    viewer.add_image(
        _downsample(v0, 2),
        name="v0",
        blending="additive",
        colormap="blue",
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        _downsample(warped_v0, 2),
        name="warped",
        blending="additive",
        colormap="green",
        scale=scale,
        visible=True,
    )
    viewer.add_image(
        _downsample(v1, 2),
        name="v1",
        blending="additive",
        colormap="red",
        scale=scale,
        visible=True,
    )
    napari.run()


if __name__ == "__main__":
    main()
