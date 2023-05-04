import logging
import math as m
from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from numpy.typing import ArrayLike

from dexpv2.constants import DEXPV2_DEBUG

LOG = logging.getLogger(__name__)


def _interpolate(tensor: th.Tensor, *args, **kwargs) -> th.Tensor:
    mode = "trilinear" if tensor.ndim == 5 else "bilinear"
    return F.interpolate(tensor, *args, **kwargs, mode=mode, align_corners=False)


def total_variation_loss(tensor: th.Tensor) -> th.Tensor:
    loss = 0.0
    for i in range(1, tensor.ndim - 1):
        idx = th.arange(tensor.shape[i], device=tensor.device)
        # tv = th.square(
        #     2 * th.index_select(tensor, i, idx[1:-1]) \
        #       - th.index_select(tensor, i, idx[2:]) \
        #       - th.index_select(tensor, i, idx[:-2])
        # )  # second derivative
        tv = th.square(
            th.index_select(tensor, i, idx[:-1]) - th.index_select(tensor, i, idx[1:])
        )  # first derivative

        loss = loss + tv.sum()
    return loss


def identity_grid(shape: tuple[int, ...]) -> th.Tensor:
    """Grid equivalent to a identity vector field (no flow).

    Parameters
    ----------
    shape : tuple[int, ...]
        Grid shape.

    Returns
    -------
    th.Tensor
        Tensor of shape (Z, Y, X, D)
    """
    ndim = len(shape)
    T = th.zeros((1, ndim, ndim + 1))
    T[:, :, :-1] = th.eye(ndim)
    grid_shape = (1, 1) + shape
    return F.affine_grid(T, grid_shape, align_corners=False)


def flow_field(
    source: th.Tensor,
    target: th.Tensor,
    im_factor: int = 4,
    grid_factor: int = 4,
    num_iterations: int = 2000,
    lr: float = 1e-4,
) -> th.Tensor:
    """
    Compute the flow vector field `T` that minimizes the
    mean squared error between `T(source)` and `target`.

    Parameters
    ----------
    source : torch.Tensor
        Source image (C, Z, Y, X).
    target : torch.Tensor
        Target image (C, Z, Y, X).
    im_factor : int, optional
        Image space down scaling factor, by default 4.
    grid_factor : int, optional
        Grid space down scaling factor, by default 4.
        Grid dimensions will be divided by both `im_factor` and `grid_factor`.
    num_iterations : int, optional
        Number of gradient descent iterations, by default 2000.
    lr : float, optional
        Learning rate (gradient descent step), by default 1e-4

    Returns
    -------
    torch.Tensor
        Vector field array with shape (D, (Z / factor), Y / factor, X / factor)
    """
    assert source.shape == target.shape
    assert source.ndim == 4

    ndim = source.ndim - 1

    source = source.unsqueeze(0)
    target = target.unsqueeze(0)

    device = source.device
    kwargs = dict(device=device, requires_grad=False)

    source = _interpolate(source, scale_factor=1 / im_factor)
    target = _interpolate(target, scale_factor=1 / im_factor)

    LOG.info(f"source / target shape: {source.shape}")

    grid_shape = tuple(m.ceil(s / grid_factor) for s in source.shape[-3:])
    grid0 = identity_grid(grid_shape).to(device)

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

        im2hat = F.grid_sample(target, large_grid, align_corners=False)
        loss = F.l1_loss(im2hat, source)
        loss = loss + total_variation_loss(grid - grid0)
        loss.backward()

        LOG.info(f"iter. {i} MSE: {loss:0.4f}")

        grid = grid - lr * grid.grad
        grid.requires_grad_(True).retain_grad()

    with th.no_grad():
        grid = grid - grid0
        grid = th.flip(grid, (-1,))  # x, y, z -> z, y, x

        # divided by 2.0 because the range is -1 to 1 (length = 2.0)
        grid = grid * im_factor * th.tensor(source.shape[-ndim:], **kwargs) / 2.0

        grid = th.stack(
            [
                _interpolate(grid[None, ..., d], source.shape[-3:])[0]
                for d in range(grid.shape[-1])
            ],
            dim=1,
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


def apply_field(field: th.Tensor, image: th.Tensor) -> th.Tensor:
    """
    Transform image using vector field.
    Image will be scaled to the field size.

    Parameters
    ----------
    field : th.Tensor
        Vector field (D, z, y, x)
    image : th.Tensor
        Original image used to compute the vector field.

    Returns
    -------
    th.Tensor
        Transformed image (z, y, x)
    """
    assert image.ndim == 4

    field = th.flip(field, (0,))  # z, y, x -> x, y, z
    field = field.movedim(0, -1)[None]

    shape = th.tensor(image.shape[::-1], device=field.device)
    field = field / shape[:-1] * 2.0  # mapping range from image shape to -1 to 1
    field = identity_grid(field.shape[1:-1]).to(field.device) - field

    transformed_image = F.grid_sample(image[None], field, align_corners=False)

    return transformed_image[0]


def advenct_field(
    field: ArrayLike,
    sources: th.Tensor,
    shape: Optional[tuple[int, ...]] = None,
    invert: bool = False,
) -> th.Tensor:
    """
    Advenct points from sources through the provided field.
    Shape indicates the original shape (space) and sources.
    Useful when field is down scaled from the original space.

    Parameters
    ----------
    field : ArrayLike
        Field array with shape T x D x (Z) x Y x X
    sources : th.Tensor
        Array of sources N x D
    shape : tuple[int, ...]
        When provided scales field accordingly, D-dimensional tuple.
    invert : bool
        When true flow is multiplied by -1, resulting in reversal of the flow.

    Returns
    -------
    th.Tensor
        Trajectories of sources N x T x D
    """
    ndim = field.ndim - 2
    device = sources.device
    orig_shape = th.tensor(shape, device=device)
    field_shape = th.tensor(field.shape[2:], device=device)

    if orig_shape is None:
        scales = th.ones(ndim, device=device)
    else:
        scales = (field_shape - 1) / (orig_shape - 1)

    trajectories = [sources]

    zero = th.zeros(1, device=device)

    for t in range(field.shape[0]):
        current = th.as_tensor(field[t]).to(device=device, non_blocking=True)

        int_sources = th.round(trajectories[-1] * scales)
        int_sources = th.maximum(int_sources, zero)
        int_sources = th.minimum(int_sources, field_shape - 1).int()
        spatial_idx = tuple(
            t.T[0] for t in th.tensor_split(int_sources, len(orig_shape), dim=1)
        )
        idx = (slice(None), *spatial_idx)

        if invert:
            sources = sources - current[idx].T
        else:
            sources = sources + current[idx].T

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
