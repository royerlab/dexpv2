import logging
import math as m
from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from numpy.typing import ArrayLike
from toolz import curry

from dexpv2.constants import DEXPV2_DEBUG

LOG = logging.getLogger(__name__)


def _interpolate(tensor: th.Tensor, *args, **kwargs) -> th.Tensor:
    mode = "trilinear" if tensor.ndim == 5 else "bilinear"
    return F.interpolate(tensor, *args, **kwargs, mode=mode, align_corners=False)


def _cumsum(tensor: th.Tensor) -> th.Tensor:
    tensor = tensor.double()
    for i in range(2, tensor.ndim):
        tensor = tensor.cumsum(dim=i)
    return tensor


def earth_mover_loss(input: th.Tensor, target: th.Tensor) -> th.Tensor:
    input = _cumsum(input)
    target = _cumsum(target)
    loss = th.abs(input - target).mean()
    return loss


def total_variation_loss(tensor: th.Tensor) -> th.Tensor:
    loss = 0.0
    for i in range(1, tensor.ndim - 1):
        idx = th.arange(tensor.shape[i], device=tensor.device)
        tv = th.square(
            th.index_select(tensor, i, idx[1:]) - th.index_select(tensor, i, idx[:-1])
        )
        loss = loss + tv.sum()
    return loss


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

    avg_kernel = th.ones((3,) * ndim, device=device)[None, None, ...]
    avg_kernel = avg_kernel / avg_kernel.sum()
    avg_filter = curry(th.conv3d, weight=avg_kernel, padding=1)

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
                # _interpolate(avg_filter(grid[None, ..., d]), source.shape[-3:])[0]
                for d in range(grid.shape[-1])
            ],
            dim=-1,
        )

        im2hat = F.grid_sample(source, large_grid)
        # loss = F.mse_loss(im2hat, target)
        # loss = earth_mover_loss(im2hat, target)
        loss = th.maximum(
            F.mse_loss(im2hat, target, reduce=False), th.ones(1, device=device)
        ).mean()
        loss = loss + total_variation_loss(grid - grid0)
        loss.backward()

        LOG.info(f"iter. {i} MSE: {loss:0.4f}")

        grid = grid - lr * grid.grad
        grid.requires_grad_(True).retain_grad()

    with th.no_grad():
        grid = grid - grid0

        # divided by 2.0 because grid spans -1 to 1 (lenth = 2.0)
        grid = grid * im_factor * th.tensor(source.shape[-ndim:], **kwargs) / 2.0
        grid = th.moveaxis(grid, -1, 1)

        grid = th.stack(
            [avg_filter(grid[None, ..., d])[0] for d in range(grid.shape[-1])], dim=-1
        )[0]

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


def advenct_field(
    field: th.Tensor,
    sources: th.Tensor,
    shape: Optional[tuple[int, ...]] = None,
) -> th.Tensor:
    """
    Advenct points from sources through the provided field.
    Shape indicates the original shape (space) and sources.
    Useful when field is down scaled from the original space.

    Parameters
    ----------
    field : th.Tensor
        Field array with shape T x D x (Z) x Y x X
    sources : th.Tensor
        Array of sources N x D
    shape : tuple[int, ...]
        When provided scales field accordingly, D-dimensional tuple.

    Returns
    -------
    th.Tensor
        Trajectories of sources N x T x D
    """
    ndim = field.ndim - 2
    device = field.device
    orig_shape = th.tensor(shape, device=device)
    field_shape = th.tensor(field.shape[2:], device=device)

    if orig_shape is None:
        scales = th.ones(ndim, device=device)
    else:
        scales = (field_shape - 1) / (orig_shape - 1)

    trajectories = [sources]

    zero = th.zeros(1, device=device)

    for t in range(field.shape[0]):
        int_sources = th.round(trajectories[-1] * scales)
        int_sources = th.maximum(int_sources, zero)
        int_sources = th.minimum(int_sources, field_shape - 1).int()
        spatial_idx = tuple(
            t[0] for t in th.tensor_split(int_sources, len(orig_shape), dim=1)
        )
        idx = (
            t,
            slice(None),
        ) + spatial_idx
        sources = sources + field[idx].T
        print(field[idx].T)
        trajectories.append(sources)

    trajectories = th.stack(trajectories, dim=1)

    return trajectories


def to_tracks(trajectories: th.Tensor) -> np.ndarray:
    """Converts trajectories to napari tracks format.

    Parameters
    ----------
    trajectories : th.Tensor
        Input N x T x D trajectories.

    Returns
    -------
    np.ndarray
        Napari tracks (N x T) x (2 + D) array.
    """
    trajectories = trajectories.cpu().numpy()
    N, T, D = trajectories.shape

    track_ids = np.repeat(np.arange(N), T)[..., None]
    time_pts = np.tile(np.arange(T), N)[..., None]
    coordinates = trajectories.reshape(-1, D)

    return np.concatenate((track_ids, time_pts, coordinates), axis=1)
