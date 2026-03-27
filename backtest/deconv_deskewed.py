import os
from pathlib import Path

import cupy as cp
import napari
import numpy as np
import tifffile

from dexpv2.cuda import to_numpy, unified_memory
from dexpv2.deconv import lucy_richardson


def main() -> None:
    data_dir = Path(os.environ["BACKTEST_DIR"]) / "daxi_deconv"
    scale = np.asarray((0.8768124086713188, 0.439, 0.439))
    slicing = (slice(960, 1200), slice(980, 1420), slice(1240, 1560))

    viewer = napari.Viewer()

    kwargs = dict(
        blending="additive",
        scale=scale,
    )

    with unified_memory():
        for i in range(2):
            print(f"Loading {i}")
            im = tifffile.imread(data_dir / f"v{i}.tif")
            im = im[slicing]
            psf = np.load(data_dir / f"psf_{i}.npy")

            print("Deconvolving ...")
            deconved = lucy_richardson(
                cp.asarray(im, dtype=np.float32),
                cp.asarray(psf, dtype=np.float32),
                iterations=5,
                accelerated=True,
            )
            print("Done!")
            deconved = to_numpy(deconved)

            viewer.add_image(im, **kwargs, name=f"view {i}", colormap="red")
            viewer.add_image(deconved, **kwargs, name=f"deconved {i}", colormap="green")
            viewer.add_image(psf, **kwargs, name=f"psf {i}", colormap="magma")

    napari.run()


if __name__ == "__main__":
    main()
