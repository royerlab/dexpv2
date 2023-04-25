import torch as th
from toolz import curry

from dexpv2.cuda import torch_default_device
from dexpv2.flow_field import advenct_field, flow_field, to_tracks


def test_flow_field(interactive_test: bool) -> None:

    device = torch_default_device()
    intensity = 1_000
    ndim = 3
    size = (64,) * ndim
    sigma = 15
    im_factor = 2
    grid_factor = 4

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

    _flow_field = curry(
        flow_field,
        im_factor=im_factor,
        grid_factor=grid_factor,
        num_iterations=2000,
        lr=1e-4,
    )

    fields = th.stack(
        [
            _flow_field(frames[i - 1, None], frames[i, None])
            for i in range(1, len(frames))
        ]
    )

    trajectory = advenct_field(fields, mus[None, 0], size)
    tracks = to_tracks(trajectory)

    if interactive_test:
        import napari

        kwargs = {"blending": "additive", "interpolation3d": "nearest", "rgb": False}

        viewer_1 = napari.Viewer()
        # viewer.add_image(fields.cpu().numpy(), colormap="turbo", scale=(2, 2, 2))
        viewer_1.add_image(frames.cpu().numpy(), **kwargs)
        viewer_1.add_tracks(tracks)

        viewer_2 = napari.Viewer()
        viewer_2.add_image(
            th.clamp_min(fields, 0).cpu().numpy(),
            colormap="red",
            scale=(im_factor,) * 3,
            **kwargs,
        )
        viewer_2.add_image(
            th.clamp_min(-fields, 0).cpu().numpy(),
            colormap="blue",
            scale=(im_factor,) * 3,
            **kwargs,
        )

        napari.run()

    assert th.allclose(trajectory, mus.float(), atol=0.5, rtol=0.0)
