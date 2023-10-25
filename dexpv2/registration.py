import logging
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F

from dexpv2.constants import DEXPV2_DEBUG
from dexpv2.cuda import import_module

logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy and scipy.")


def _interpolate(tensor: th.Tensor, antialias: bool = False, **kwargs) -> th.Tensor:
    """Interpolates tensor.

    Parameters
    ----------
    tensor : th.Tensor
        Input 4 or 5-dim tensor.
    antialias : bool, optional
        When true applies a gaussian filter with 0.5 * downscale factor.
        Ignored if upscaling.

    Returns
    -------
    th.Tensor
        Interpolated tensor.
    """
    mode = "trilinear" if tensor.ndim == 5 else "bilinear"
    if antialias:
        scale_factor = kwargs.get("scale_factor")
        if scale_factor is None:
            raise ValueError(
                "`_interpolate` with `antialias=True` requires `scale_factor` parameter."
            )

        if scale_factor < 1.0:
            ndi = import_module("scipy", "ndimage")
            orig_shape = tensor.shape
            array = xp.asarray(tensor.squeeze())
            blurred = ndi.gaussian_filter(
                array,
                sigma=0.5 / scale_factor,
                output=array,
            )
            tensor = th.as_tensor(blurred, device=tensor.device)
            tensor = tensor.reshape(orig_shape)
            LOG.info(f"Antialiasing with sigma = {0.5 / scale_factor}.")

    return F.interpolate(tensor, **kwargs, mode=mode, align_corners=True)


class Registration:
    def __init__(
        self,
        im_factor: int = 4,
        grid_factor: int = 4,
        num_iterations: int = 1000,
        lr: float = 1e-4,
        n_scales: int = 3,
    ) -> None:
        """
        Registration class.

        Parameters
        ----------
        im_factor : int, optional
            Image space down scaling factor, by default 4.
        grid_factor : int, optional
            Grid space down scaling factor, by default 4.
            Grid dimensions will be divided by both `im_factor` and `grid_factor`.
        num_iterations : int, optional
            Number of gradient descent iterations, by default 1000.
        lr : float, optional
            Learning rate (gradient descent step), by default 1e-4
        n_scales : int, optional
            Number of scales used for multi-scale optimization, by default 3.
        """

        self.im_factor = im_factor
        self.grid_factor = grid_factor
        self.num_iterations = num_iterations
        self.lr = lr
        self.n_scales = n_scales

    @abstractmethod
    def zero_grad(self) -> None:
        # TODO
        raise NotImplementedError

    @abstractmethod
    def grad_step(self) -> None:
        # TODO
        raise NotImplementedError

    @abstractmethod
    def setup_params(self, reference_tensor: th.Tensor) -> None:
        # TODO
        raise NotImplementedError

    def regularization_loss(self) -> th.Tensor | float:
        return 0

    @abstractmethod
    def apply(self, tensor: th.Tensor) -> th.Tensor:
        # TODO
        raise NotImplementedError

    @abstractmethod
    def formated_grid(self) -> th.Tensor:
        # TODO
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        # TODO
        raise NotImplementedError

    def fit(
        self,
        source: th.Tensor,
        target: th.Tensor,
    ) -> "Registration":
        """
        Compute the flow vector field `T` that minimizes the
        mean squared error between `T(source)` and `target`.

        Parameters
        ----------
        source : torch.Tensor
            Source image (C, Z, Y, X).
        target : torch.Tensor
            Target image (C, Z, Y, X).

        """
        self.reset()

        assert source.shape == target.shape
        assert source.ndim == 4
        assert self.n_scales > 0

        source = source.unsqueeze(0)
        target = target.unsqueeze(0)

        scales = np.flip(np.power(2, np.arange(self.n_scales)))

        for scale in scales:
            with th.no_grad():
                scaled_im_factor = self.im_factor * scale
                scaled_source = _interpolate(
                    source, scale_factor=1 / scaled_im_factor, antialias=True
                )
                scaled_target = _interpolate(
                    target, scale_factor=1 / scaled_im_factor, antialias=True
                )
                self.setup_params(scaled_source)

            LOG.info(f"scale: {scale}")
            LOG.info(f"image shape: {scaled_source.shape}")

            for i in range(self.num_iterations):
                self.zero_grad()

                im2hat = self.apply(target)

                loss = F.l1_loss(im2hat, scaled_source) + self.regularization_loss()

                loss.backward()

                if i % 10 == 0:
                    LOG.info(f"iter. {i} MSE: {loss:0.4f}")

                self.grad_step()

        LOG.info(f"image size: {source.shape}")
        LOG.info(f"image factor: {self.im_factor}")
        LOG.info(f"grid factor: {self.grid_factor}")

        if DEXPV2_DEBUG:
            import napari

            viewer = napari.Viewer()
            viewer.add_image(
                scaled_source.cpu().numpy(),
                name="im1",
                blending="additive",
                colormap="blue",
                visible=False,
            )
            viewer.add_image(
                scaled_target.cpu().numpy(),
                name="im2",
                blending="additive",
                colormap="green",
            )
            viewer.add_image(
                im2hat.detach().cpu().numpy(),
                name="im2hat",
                blending="additive",
                colormap="red",
            )
            napari.run()

        return self


class AffineRegistration(Registration):
    def __init__(self, init_affine: Optional[th.Tensor] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.init_affine = init_affine

    def zero_grad(self) -> None:
        # TODO
        if self.affine.grad is not None:
            self.affine.grad.zero_()

    def grad_step(self) -> None:
        # TODO
        with th.no_grad():
            self.affine -= self.lr * self.affine.grad

    def setup_params(self, reference_tensor: th.Tensor) -> None:
        # TODO
        if not hasattr(self, "affine"):
            ndim = reference_tensor.ndim - 2
            if ndim not in (2, 3):
                raise ValueError(f"Image dimensions must be 2 or 3, got {ndim}.")

            shape = (1, ndim, ndim + 1)

            if self.init_affine is None:
                self.affine = th.zeros(shape)
                self.affine[:, :, :-1] = th.eye(ndim)
                self.init_affine = self.affine.clone()
            else:
                try:
                    self.affine = self.init_affine.clone().reshape(shape)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid affine shape {self.init_affine.shape}, must be compatible to reshape to {shape}."
                    ) from e

            self.init_affine = self.init_affine.to(reference_tensor.device)
            self.affine = self.affine.to(reference_tensor.device)
            self.affine.requires_grad_(True).retain_grad()

        # changes with every scale
        self.reference_shape = reference_tensor.shape

    def regularization_loss(self) -> th.Tensor | float:
        return 0.0  # return 0.01 * th.norm(self.affine[:, :, :-1] - self.init_affine[:, :, :-1], p=2)

    def apply(self, tensor: th.Tensor, keep_shape: bool = False) -> th.Tensor:
        # TODO
        if keep_shape:
            shape = tensor.shape[-3:]
        else:
            shape = self.reference_shape[-3:]

        grid = F.affine_grid(self.affine, size=(1, 1) + shape, align_corners=True)
        return F.grid_sample(tensor, grid, align_corners=True)

    def formated_grid(self) -> th.Tensor:
        pass

    def reset(self) -> None:
        # TODO
        if hasattr(self, "affine"):
            del self.affine


if __name__ == "__main__":
    from pathlib import Path

    import napari
    import tifffile

    device = "cuda:1"
    root = Path("/mnt/md0/daxi-debugging")
    scale = np.asarray((0.8768124086713188, 0.439, 0.439))

    v0 = tifffile.imread(root / "v0.tif")[0].astype(np.float32)
    v1 = tifffile.imread(root / "v1.tif")[0].astype(np.float32)

    v0 = np.clip(v0, 100, None) - 120
    v1 = np.clip(v1, 100, None) - 120

    v0 = v0[::2, ::2, ::2]
    v1 = v1[::2, ::2, ::2]

    v0 /= v0.max()
    v1 /= v1.max()

    th_v1 = th.as_tensor(v1[None, ...], device=device)
    th_v0 = th.as_tensor(v0[None, ...], device=device)

    model = AffineRegistration(num_iterations=3000, lr=0.1)
    model.fit(th_v0, th_v1)

    with th.no_grad():
        reg = model.apply(th_v1[None, ...], keep_shape=True).cpu().numpy()
        print(model.affine)

    viewer = napari.Viewer()

    viewer.add_image(v0, name="v0", blending="additive", colormap="red", scale=scale)
    viewer.add_image(
        v1, name="v1", blending="additive", colormap="green", scale=scale, visible=False
    )
    viewer.add_image(
        reg,
        name="reg",
        blending="additive",
        colormap="blue",
        scale=scale,
    )

    napari.run()
