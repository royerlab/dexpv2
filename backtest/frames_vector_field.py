import os
from pathlib import Path

import click
import torch as th
from tifffile import imread

from dexpv2.cli.utils import interactive_option, log_level_option
from dexpv2.cuda import torch_default_device
from dexpv2.vector_field import apply_field, vector_field


def _load_tensor(path: Path) -> th.Tensor:
    device = torch_default_device()
    return th.tensor(imread(path).astype("float32"), device=device)


@click.command()
@log_level_option()
@interactive_option()
def main(interactive: bool) -> None:
    """Runs vector field estimation in two frames."""

    data_dir = Path(os.environ["BACKTEST_DIR"]) / "frames_vector_field"
    assert data_dir.exists()

    im_factor = 4

    im1 = _load_tensor(data_dir / "450.tif")
    im2 = _load_tensor(data_dir / "455.tif")

    field = vector_field(im1, im2, im_factor=im_factor)
    im2hat = apply_field(field, im1)
    scale = (im_factor,) * im1.ndim

    if interactive:
        import napari

        viewer = napari.Viewer()
        kwargs = {"blending": "additive"}

        viewer.add_image(im1.cpu().numpy(), colormap="blue", name="frame 450", **kwargs)
        viewer.add_image(
            im2.cpu().numpy(), colormap="green", name="frame 455", **kwargs
        )
        viewer.add_image(
            im2hat.cpu().numpy(),
            colormap="red",
            name="T(frame 450)",
            scale=scale,
            **kwargs,
        )
        viewer.add_image(
            field.cpu().numpy(),
            name="vector field",
            visible=False,
            colormap="turbo",
            scale=scale,
            **kwargs,
        )

        napari.run()


if __name__ == "__main__":
    main()
