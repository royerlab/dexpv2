import itertools
import os
from pathlib import Path
from typing import Optional, Tuple

import cupy as cp
import napari
import numpy as np
import tifffile

from dexpv2.crop import foreground_bbox
from dexpv2.cuda import unified_memory


def _downsample(
    arr: np.ndarray, step: int, axis: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    if axis is None:
        axis = tuple(range(arr.ndim))

    slicing = [slice(None)] * arr.ndim
    for i in axis:
        slicing[i] = slice(None, None, step)

    return arr[tuple(slicing)]


def to_rectangle(bbox: np.ndarray) -> np.ndarray:
    bbox = bbox.reshape(2, -1).T
    rect = np.array(tuple(itertools.product(*bbox)))
    return rect


def main() -> None:
    data_dir = Path(os.environ["BACKTEST_DIR"]) / "daxi_warp_registration"

    scale = np.asarray([0.8768124086713188, 0.439, 0.439])

    v0 = tifffile.imread(data_dir / "v0.tif")
    v1 = tifffile.imread(data_dir / "v1.tif")

    with unified_memory():
        bbox0 = foreground_bbox(cp.asarray(v0), scale, downscale=16)
        bbox1 = foreground_bbox(cp.asarray(v1), scale, downscale=16)

    viewer = napari.Viewer()

    viewer.add_image(
        _downsample(v0, 4),
        name="v0",
        blending="additive",
        colormap="green",
        scale=scale,
        visible=False,
    )
    viewer.add_points(
        to_rectangle(bbox0) / 4,
        name="bbox0",
        edge_color="green",
        face_color="green",
        scale=scale,
        visible=True,
    )
    viewer.add_image(
        _downsample(v1, 4),
        name="v1",
        blending="additive",
        colormap="red",
        scale=scale,
        visible=True,
    )
    viewer.add_points(
        to_rectangle(bbox1) / 4,
        name="bbox1",
        edge_color="red",
        face_color="red",
        scale=scale,
        visible=True,
    )
    napari.run()


if __name__ == "__main__":
    main()
