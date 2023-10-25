import logging
import math as m
from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from numpy.typing import ArrayLike

from dexpv2.registration import Registration, _interpolate

LOG = logging.getLogger(__name__)


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

        loss = loss + tv.mean()
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
    grid = F.affine_grid(T, grid_shape, align_corners=True)
    return grid


def _interpolate_grid(grid: th.Tensor, out_dim: int = -1, **kwargs) -> th.Tensor:
    """Interpolate a grid.

    Parameters
    ----------
    grid : th.Tensor
        Grid tensor of shape (z, y, x, D).
    out_dim : int, optional
        Output dimension, by default -1.

    Returns
    -------
    th.Tensor
        Tensor of shape (Z, Y, X, D) when out_dim=1, D position is subject to `out_dim` value.
    """
    return th.stack(
        [_interpolate(grid[None, ..., d], **kwargs)[0] for d in range(grid.shape[-1])],
        dim=out_dim,
    )


class FlowFieldRegistration(Registration):
    def zero_grad(self) -> None:
        # TODO
        if self.grid.grad is not None:
            self.grid.grad.zero_()

    def grad_step(self) -> None:
        # TODO
        with th.no_grad():
            self.grid -= self.lr * self.grid.grad

    def setup_params(self, reference_tensor: th.Tensor) -> None:
        # TODO
        with th.no_grad():
            if not hasattr(self, "grid"):
                grid_shape = tuple(
                    m.ceil(s / self.grid_factor) for s in reference_tensor.shape[-3:]
                )
                self.grid = identity_grid(grid_shape).to(reference_tensor.device)
                self.grid0 = self.grid.clone()
            else:
                self.grid = _interpolate_grid(self.grid, scale_factor=2)
                self.grid0 = identity_grid(self.grid.shape[-4:-1]).to(
                    reference_tensor.device
                )

        self.reference_shape = reference_tensor.shape

        self.grid.requires_grad_(True).retain_grad()

    def regularization_loss(self) -> th.Tensor | float:
        # TODO
        return total_variation_loss(self.grid - self.grid0)

    def apply(self, tensor: th.Tensor) -> th.Tensor:
        # TODO
        large_grid = _interpolate_grid(self.grid, size=self.reference_shape[-3:])
        return F.grid_sample(tensor, large_grid, align_corners=True)

    def formated_grid(self) -> th.Tensor:

        # TODO
        """_summary_


        Returns
        -------
        torch.Tensor
            Vector field array with shape (D, (Z / factor), Y / factor, X / factor)
        """
        with th.no_grad():
            grid = self.grid - self.grid0
            grid = th.flip(grid, (-1,))  # x, y, z -> z, y, x

            # divided by 2.0 because the range is -1 to 1 (length = 2.0)
            grid /= 2.0
            grid = _interpolate_grid(grid, out_dim=1, size=self.reference_shape[-3:])[0]

            LOG.info(f"vector field shape: {grid.shape}")
            return grid.cpu()

    def reset(self) -> None:
        # TODO
        if hasattr(self, "grid"):
            del self.grid
            del self.grid0


@th.no_grad()
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

    field = field * 2.0  # mapping range from image shape to -1 to 1
    field = identity_grid(field.shape[1:-1]).to(field.device) - field

    transformed_image = F.grid_sample(image[None], field, align_corners=True)

    return transformed_image[0]


@th.no_grad()
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

        movement = current[idx].T * orig_shape

        if invert:
            sources = sources - movement
        else:
            sources = sources + movement

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
