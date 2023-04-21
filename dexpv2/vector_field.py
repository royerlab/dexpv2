import logging
import math as m

import torch as th
import torch.nn.functional as F
from numpy.typing import ArrayLike

from dexpv2.constants import DEXPV2_DEBUG

LOG = logging.getLogger(__name__)


def _interpolate(tensor: th.Tensor, *args, **kwargs) -> th.Tensor:
    mode = "triliear" if tensor.ndim == 5 else "bilinear"
    return F.interpolate(tensor, *args, **kwargs, mode=mode, align_corners=False)


def vector_field(
    source: ArrayLike,
    target: ArrayLike,
    im_factor: int = 8,
    grid_factor: int = 4,
    num_iterations: int = 1500,
    lr: float = 1e-4,
) -> ArrayLike:
    """
    Compute the vector field `T` that minimizes the
    mean squared error between `T(source)` and `target`.

    Parameters
    ----------
    source : ArrayLike
        Source image.
    target : ArrayLike
        Target image.
    im_factor : int, optional
        Image space down scaling factor, by default 8.
    grid_factor : int, optional
        Grid space down scaling factor, by default 4.
        Grid dimensions will be divided by both `im_factor` and `grid_factor`.
    num_iterations : int, optional
        Number of gradient descent iterations, by default 1500.
    lr : float, optional
        Learning rate (gradient descent step), by default 1e-4

    Returns
    -------
    ArrayLike
        Vector field array with shape (D, (Z / factor), Y / factor, X / factor)
    """
    ndim = source.ndim

    source = th.as_tensor(source[None, None])
    target = th.as_tensor(target[None, None])

    device = source.device
    kwargs = dict(device=device, requires_grad=False)

    source = _interpolate(source, scale_factor=1 / im_factor)
    target = _interpolate(target, scale_factor=1 / im_factor)

    LOG.info(f"source / target shape: {source.shape}")

    T = th.zeros((1, ndim, ndim + 1))
    T[:, :, :-1] = th.eye(ndim)

    grid_shape = (1, 1) + tuple(m.ceil(s / grid_factor) for s in source.shape[-3:])
    grid0 = F.affine_grid(T.to(device), grid_shape)

    grid = grid0.detach()
    grid.requires_grad_(True).retain_grad()

    LOG.info(f"grid shape: {grid.shape}")

    for i in range(num_iterations):
        if grid.grad is not None:
            grid.grad.zero_()

        large_grid = th.stack(
            [
                _interpolate(grid[None, ..., d], source.shape[-3:])[0]
                for d in range(grid.shape[-1])
            ],
            dim=-1,
        )

        im2hat = F.grid_sample(source, large_grid)
        loss = F.mse_loss(im2hat, target)
        loss.backward()

        LOG.info(f"iter. {i} MSE: {loss:0.4f}")

        grid = grid - lr * grid.grad
        grid.requires_grad_(True).retain_grad()

    with th.no_grad():
        grid = grid - grid0

        # divided by 2.0 because grid spans -1 to 1 (lenth = 2.0)
        grid = grid * im_factor * th.tensor(source.shape[-ndim:], **kwargs) / 2.0
        grid = th.moveaxis(grid, -1, 1)

        # mean filter
        avg = th.zeros((ndim, ndim, 3, 3, 3), **kwargs)
        avg[th.arange(ndim), th.arange(ndim)] = 1 / 27
        grid = F.conv3d(grid, avg, padding=1).detach()

        LOG.info(f"vector field shape: {grid.shape}")

    if DEXPV2_DEBUG:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(
            source.cpu().numpy(), name="im1", blending="additive", colormap="blue"
        )
        viewer.add_image(
            target.cpu().numpy(), name="im2", blending="additive", colormap="green"
        )
        viewer.add_image(
            im2hat.detach().cpu().numpy(),
            name="im2hat",
            blending="additive",
            colormap="red",
        )
        viewer.add_image(
            grid.detach().cpu().numpy(),
            name="grid",
            scale=(grid_factor,) * ndim,
            colormap="turbo",
        )
        napari.run()

    return grid
