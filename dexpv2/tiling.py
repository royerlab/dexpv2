import logging
from itertools import product
from typing import Callable, Protocol, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from dexpv2.cuda import to_numpy

LOG = logging.getLogger(__name__)


class BlendingMap:
    """
    Class that applies blending to tiles.

    Parameters
    ----------
    tile : Tuple[int, ...]
        Sizes of the tiles along each dimension.
    overlap : Tuple[int, ...]
        Overlaps of tiles along each dimension.
    num_non_tiled : int
        Number of dimensions that are not tiled.
    to_device : Callable[[ArrayLike], ArrayLike], optional
        Function to send tiles to device expected by `func`, by default None.
    """

    def __init__(
        self,
        tile: Tuple[int, ...],
        overlap: Tuple[int, ...],
        num_non_tiled: int,
        to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
    ) -> None:

        self.tile = tile
        self.overlap = overlap
        self.num_non_tiled = num_non_tiled
        self.to_device = to_device

        self.blending_map = self._create_blending_map(
            tile, overlap, to_device=to_device
        )

    @staticmethod
    def _create_blending_map(
        tile: Tuple[int, ...],
        overlap: Tuple[int, ...],
        ignore_right: Tuple[int, ...] = tuple(),
        to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
    ) -> ArrayLike:
        """
        Generate a blending map for tiles with specified overlaps.

        Parameters
        ----------
        tile : Tuple[int, ...]
            Sizes of the tiles along each dimension.
        overlap : Tuple[int, ...]
            Overlaps of tiles along each dimension.
        ignore_right : Tuple[int, ...], optional
            Indices of dimensions to ignore right-side blending, by default an empty tuple.
        to_device : Callable[[ArrayLike], ArrayLike], optional
            Function to send tiles to device expected by `func`, by default None.

        Returns
        -------
        np.ndarray
            Blending map for tiles with specified overlaps.
        """
        blending_map = 1
        for i in range(len(tile)):
            if 2 * overlap[i] > tile[i]:
                raise ValueError(
                    f"2x overlap {overlap[i]} cannot be larger than tile size {tile[i]}."
                )
            left_border = np.linspace(0, 1, overlap[i] + 2)[1:-1]
            if i in ignore_right:
                right_border = np.ones_like(left_border)
            else:
                right_border = np.ones_like(left_border) - left_border
            line_blending = np.concatenate(
                (left_border, np.ones(tile[i]), right_border)
            )
            line_blending = line_blending[
                (None,) * i + (...,) + (None,) * (len(tile) - i - 1)
            ]
            blending_map = blending_map * line_blending

        return to_device(blending_map)

    def __call__(self, array: ArrayLike) -> ArrayLike:
        """Applies blending coefficients to a given array.

        Parameters
        ----------
        array : ArrayLike
            Input array to be blended.

        Returns
        -------
        ArrayLike
            Blended array.
        """
        short_axes = tuple(
            i
            for i in range(len(self.tile))
            if array.shape[i + self.num_non_tiled] != self.blending_map.shape[i]
        )

        if len(short_axes) > 0:
            # ignoring right side blending on axes that the last tile doesn't overhang
            fixed_blending = self.to_device(
                self._create_blending_map(
                    self.tile,
                    self.overlap,
                    ignore_right=short_axes,
                )
            )
            array = (
                array
                * fixed_blending[
                    (...,) + tuple(slice(s) for s in array.shape[self.num_non_tiled :])
                ]
            )
        else:
            array = array * self.blending_map

        return array


def apply_tiled(
    arr: np.ndarray,
    func: Callable[[ArrayLike], ArrayLike],
    tile: Tuple[int, ...],
    overlap: Union[int, Tuple[int, ...]],
    pad: str = "reflect",
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
    out_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Apply a given function to tiled portions of an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array to be tiled and processed.
    func : Callable[[ArrayLike], ArrayLike]
        Function to apply to each tile.
    tile : Tuple[int, ...]
        Sizes of the tiles along each dimension.
    overlap : Union[int, Tuple[int, ...]]
        Overlaps of tiles along each dimension.
    pad : str, optional
        Padding mode for tiling, by default "reflect".
    to_device : Optional[Callable[[ArrayLike], ArrayLike]], optional
        Function to send tiles to device expected by `func`.
    out_dtype : np.dtype, optional
        Data type of the output array, by default np.float32.

    Returns
    -------
    np.ndarray
        Output array with the function applied and blended together for each tiled portions.
    """
    if isinstance(overlap, int):
        overlap = (overlap,) * len(tile)

    if len(tile) != len(overlap):
        raise ValueError(
            f"The length of tiles and overlaps must be the same. Got {len(tile)} and {len(overlap)}."
        )

    num_non_tiled = len(arr.shape) - len(tile)
    orig_shape = arr.shape[num_non_tiled:]
    pad_width = ((0, 0),) * num_non_tiled + tuple(
        (overlap, overlap) for overlap in overlap
    )

    arr = np.pad(arr, pad_width, mode=pad)
    out_arr = None

    blending = BlendingMap(tile, overlap, num_non_tiled, to_device=to_device)

    tiling_start = list(
        product(
            *[
                range(o, size + 2 * o, t + o)
                for size, t, o in zip(orig_shape, tile, overlap)
            ]
        )
    )

    for start_indices in tqdm(tiling_start, "Applying function to tiles"):
        slicing = (...,) + tuple(
            slice(start - o, start + t + o)
            for start, t, o in zip(start_indices, tile, overlap)
        )

        in_tile = to_device(arr[slicing])

        out_tile = func(in_tile)
        del in_tile

        if out_arr is None:
            out_arr = np.zeros(
                out_tile.shape[:num_non_tiled] + arr.shape[num_non_tiled:],
                dtype=out_dtype,
            )

        out_tile = blending(out_tile)
        out_tile = to_numpy(out_tile)
        out_arr[slicing] += out_tile
        del out_tile

    slicing = (...,) + tuple(slice(o, size + o) for size, o in zip(orig_shape, overlap))
    if out_arr is None:
        raise ValueError("No tiles were processed.")

    out_arr = out_arr[slicing]

    assert (
        out_arr.shape[num_non_tiled:] == orig_shape
    ), f"{out_arr.shape} != {orig_shape}"

    return out_arr


class ArrayFunc(Protocol):
    def __call__(self, *args: ArrayLike) -> ArrayLike:
        ...


def apply_tiled_stacked(
    *arrays: np.ndarray,
    func: ArrayFunc,
    tile: Tuple[int, ...],
    overlap: Union[int, Tuple[int, ...]],
    pad: str = "constant",
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
) -> np.ndarray:
    """
    Apply a given function to tiled portions of an array and stack the results.
    It's more versatile than `apply_tiled` but it uses more memory and does not blend the output tiles.

    Parameters
    ----------
    arrays : np.ndarray
        Input arrays to be tiled and processed.
    func : ArrayFunc
        Function to apply to each tile.
    tile : Tuple[int, ...]
        Sizes of the tiles along each dimension.
    overlap : Union[int, Tuple[int, ...]]
        Overlaps of tiles along each dimension.
    pad : str, optional
        Padding mode for tiling, by default "constant".
    to_device : Optional[Callable[[ArrayLike], ArrayLike]], optional
        Function to send tiles to device expected by `func`, by default None.

    Returns
    -------
    np.ndarray
        Output array with the function applied to tiled portions.
    """
    if isinstance(overlap, int):
        overlap = (overlap,) * len(tile)

    if len(tile) != len(overlap):
        raise ValueError(
            f"The length of tiles and overlaps must be the same. Got {len(tile)} and {len(overlap)}."
        )

    shape = arrays[0].shape
    num_non_tiled = len(shape) - len(tile)
    orig_shape = shape[num_non_tiled:]
    pad_width = ((0, 0),) * num_non_tiled + tuple(
        (overlap, overlap) for overlap in overlap
    )

    arrays = tuple(np.pad(a, pad_width, mode=pad) for a in arrays)

    axis_iterators = [
        range(o, size + 2 * o, t + o) for size, t, o in zip(orig_shape, tile, overlap)
    ]
    out_shape = tuple(len(i) for i in axis_iterators)
    LOG.info(f"Number of tiles per axis: {out_shape}")
    out_arrays = []

    tiling_start = list(product(*axis_iterators))

    for start_indices in tqdm(tiling_start, "Applying function to tiles"):
        slicing = (...,) + tuple(
            slice(start - o, start + t + o)
            for start, t, o in zip(start_indices, tile, overlap)
        )

        in_tiles = [to_device(a[slicing]) for a in arrays]

        out_tile = func(*in_tiles)
        LOG.info(f"Output tile shape: {out_tile.shape}")

        out_arrays.append(to_numpy(out_tile))
        del in_tiles, out_tile

    if len(out_arrays) == 0:
        raise ValueError("No tiles were processed.")

    out_arr = np.stack(out_arrays, axis=-1)
    LOG.info(f"Output stacekd array shape: {out_arr.shape}")

    out_arr = out_arr.reshape((-1, *out_shape))
    LOG.info(f"Output reshape array shape: {out_arr.shape}")

    return out_arr
