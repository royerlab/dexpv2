import os
from pathlib import Path

import click
import cupy as cp
import numpy as np
from numpy.typing import ArrayLike
from tifffile import imread

from dexpv2.cli.utils import interactive_option, log_level_option
from dexpv2.cuda import unified_memory
from dexpv2.deskew import deskew
from dexpv2.utils import to_cpu


def estimate_shift(skewed: ArrayLike, z_res: float, x_res: float) -> None:

    skewed = cp.asarray(skewed)
    skewed = skewed / skewed.max()

    freq_skewed = cp.fft.fftn(skewed, axes=(0, 2))
    freq_skewed = freq_skewed[1]
    mean_freq = cp.abs(freq_skewed.mean(axis=(0,)))
    mean_freq = cp.fft.fftshift(mean_freq)
    top_freq = cp.flip(cp.argsort(mean_freq))

    print(f"top freq: {top_freq[:11] - len(top_freq) / 2}")

    # import matplotlib.pyplot as plt
    # plt.bar(np.arange(len(top_freq)) - len(top_freq) / 2, np.log(mean_freq.get()))
    # plt.show()

    shift = 4
    real_shift = shift * x_res
    hypothenuse = np.sqrt(real_shift**2 + z_res**2)
    print(f"angle estimate = {np.rad2deg(np.arccos(real_shift / hypothenuse))}")

    # deskewed = skewed.copy()
    # for i in tqdm(range(skewed.shape[0])):
    #     deskewed[i] = cp.roll(skewed[i], i * shift, axis=-1)

    # viewer.add_image(skewed.get(), name="skewed", **kwargs)
    # viewer.add_image(deskewed.get(), name="deskewed", **kwargs)
    # napari.run()
    # return


@click.command()
@log_level_option()
@interactive_option()
def main(interactive: bool) -> None:

    data_dir = Path(os.environ["BACKTEST_DIR"]) / "daxi_deskew"
    assert data_dir.exists()

    if interactive:
        import napari

        viewer = napari.Viewer()
        kwargs = {"colormap": "magma", "blending": "additive"}

    angle = 45.0
    x_res = 0.219
    z_step = 0.155 * 5
    out_z_res = z_step * np.cos(np.deg2rad(angle))
    print(out_z_res)

    skewed = imread(data_dir / "2023_08_03.tif")[600:1000, 1000:1600, 1000:1600]
    estimate_shift(skewed, z_step, x_res)

    with unified_memory():
        deskewed = to_cpu(
            deskew(
                cp.flip(cp.asarray(skewed).transpose((0, 2, 1)), axis=1),
                px_to_scan_ratio=x_res / z_step,
                ls_angle_deg=angle,
                keep_overhang=False,
            )
        )

    cp.clear_memo()

    if interactive:
        viewer.add_image(skewed, name="skewed", **kwargs)
        viewer.add_image(deskewed, name="deskewed", **kwargs)
        napari.run()


if __name__ == "__main__":
    main()
