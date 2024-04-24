import os
import re
from pathlib import Path

import click
import cupy as cp
from tifffile import imread

from dexpv2.cli.utils import interactive_option, log_level_option
from dexpv2.crosscorr import multiview_phase_cross_corr
from dexpv2.cuda import unified_memory
from dexpv2.fusion import multiview_fuse
from dexpv2.utils import to_cpu


@click.command()
@log_level_option()
@interactive_option()
def main(interactive: bool) -> None:

    data_dir = Path(os.environ["BACKTEST_DIR"]) / "single_frame_multiview"
    assert data_dir.exists()

    if interactive:
        import napari

        viewer = napari.Viewer()
        kwargs = {"colormap": "magma", "blending": "additive"}

    channels = ["h2afva", "mezzo"]

    for ch in channels:
        views = {
            re.findall("C[01]+L[01]+", p.name)[0]: imread(p)
            for p in data_dir.glob(f"{ch}*.tif")
        }

        with unified_memory():
            translation = multiview_phase_cross_corr(
                **views, camera_1_flip=True, to_device=cp.asarray
            )

            fused = multiview_fuse(
                **views,
                camera_1_T=translation,
                camera_1_flip=True,
                to_device=cp.asarray,
            )

        if interactive:
            for name, arr in views.items():
                viewer.add_image(to_cpu(arr), name=f"{ch} {name}", **kwargs)
            viewer.add_image(to_cpu(fused), name=ch, **kwargs)

    if interactive:
        napari.run()


if __name__ == "__main__":
    main()
