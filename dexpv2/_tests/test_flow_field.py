from typing import Type

import pytest
import torch as th

from dexpv2.cuda import torch_default_device
from dexpv2.flow_field import FlowFieldRegistration, advenct_field, to_tracks
from dexpv2.registration import AffineRegistration, Registration


@pytest.mark.parametrize(
    "registration_class",
    [AffineRegistration, FlowFieldRegistration],
)
def test_flow_field(
    registration_class: Type[Registration],
    interactive_test: bool,
) -> None:

    device = torch_default_device()
    intensity = 1_000
    size = (56, 65, 72)
    sigma = 15
    im_factor = 2
    grid_factor = 4
    n_scales = 3

    grid = th.stack(
        th.meshgrid([th.arange(s, device=device) for s in size], indexing="ij"), dim=-1
    )

    mus = th.Tensor(
        [[0.5, 0.5, 0.5], [0.55, 0.5, 0.5], [0.57, 0.48, 0.53], [0.55, 0.45, 0.55]]
    ).to(device)

    mus = (mus * th.tensor(size, device=device)).round().int()

    frames = th.stack(
        [intensity * th.exp(-th.square(grid - mu).sum(dim=-1) / sigma) for mu in mus]
    )

    reg_models = [
        registration_class(
            im_factor=im_factor,
            grid_factor=grid_factor,
            num_iterations=500,
            lr=1e-4,
            n_scales=n_scales,
        ).fit(frames[i - 1, None], frames[i, None])
        for i in range(1, len(frames))
    ]

    trajectory = advenct_field(fields, mus[None, 0], size)
    tracks = to_tracks(trajectory)

    if interactive_test:
        import napari

        kwargs = {"blending": "additive", "interpolation3d": "nearest", "rgb": False}

        viewer_1 = napari.Viewer()
        viewer_1.add_image(frames.cpu().numpy(), **kwargs)
        viewer_1.add_tracks(tracks)

        napari.run()

    th.testing.assert_close(trajectory.squeeze(), mus.float(), atol=0.5, rtol=0.0)
