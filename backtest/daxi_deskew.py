import os
from pathlib import Path

import click
import cupy as cp
from tifffile import imread

from dexpv2.cli.utils import interactive_option, log_level_option
from dexpv2.deskew import deskew_data
from dexpv2.utils import to_cpu


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

    views = ["v0", "v1"]
    angle = 45.0
    x_res = 0.439
    z_step = 1.24

    for i, view_name in enumerate(views):

        skewed = imread(data_dir / f"{view_name}.tif")
        skewed = skewed[500:, :1024, 1024:]

        if i == 1:
            skewed = skewed[:, ::-1]

        deskewed = to_cpu(
            deskew_data(
                cp.asarray(skewed),
                px_to_scan_ratio=x_res / z_step,
                ls_angle_deg=angle,
                keep_overhang=False,
            )
        )

        if i == 1:
            deskewed = deskewed[::-1]

        print(skewed.shape)
        print(deskewed.shape)

        cp.clear_memo()

        if interactive:
            viewer.add_image(skewed, name=f"{view_name} skewed", **kwargs)
            viewer.add_image(deskewed, name=f"{view_name} deskewed", **kwargs)

    if interactive:
        napari.run()


if __name__ == "__main__":
    main()
