from itertools import product
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from dexpv2.cuda import to_numpy


def blending_map(
    tiles: Tuple[int, ...],
    overlaps: Tuple[int, ...],
    ignore_right: Tuple[int, ...] = tuple(),
) -> np.ndarray:
    """
    Generate a blending map for tiles with specified overlaps.

    Parameters
    ----------
    tiles : Tuple[int, ...]
        Sizes of the tiles along each dimension.
    overlaps : Tuple[int, ...]
        Overlaps of tiles along each dimension.
    ignore_right : Tuple[int, ...], optional
        Indices of dimensions to ignore right-side blending, by default an empty tuple.

    Returns
    -------
    np.ndarray
        Blending map for tiles with specified overlaps.
    """
    blending = 1
    for i in range(len(tiles)):
        if 2 * overlaps[i] > tiles[i]:
            raise ValueError(
                f"2x overlap {overlaps[i]} cannot be larger than tile size {tiles[i]}."
            )
        left_border = np.linspace(0, 1, overlaps[i] + 2)[1:-1]
        if i in ignore_right:
            right_border = np.ones_like(left_border)
        else:
            right_border = np.ones_like(left_border) - left_border
        line_blending = np.concatenate((left_border, np.ones(tiles[i]), right_border))
        line_blending = line_blending[
            (None,) * i + (...,) + (None,) * (len(tiles) - i - 1)
        ]
        blending = blending * line_blending

    return blending


def apply_tiled(
    arr: np.ndarray,
    func: Callable[[ArrayLike], ArrayLike],
    tiles: Tuple[int, ...],
    overlaps: Tuple[int, ...],
    pad: str = "reflect",
    to_device: Optional[Callable[[ArrayLike], ArrayLike]] = None,
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
    tiles : Tuple[int, ...]
        Sizes of the tiles along each dimension.
    overlaps : Tuple[int, ...]
        Overlaps of tiles along each dimension.
    pad : str, optional
        Padding mode for tiling, by default "reflect".
    to_device : Optional[Callable[[ArrayLike], ArrayLike]], optional
        Function to send tiles to device expected by `func`, by default None.
    out_dtype : np.dtype, optional
        Data type of the output array, by default np.float32.

    Returns
    -------
    np.ndarray
        Output array with the function applied to tiled portions.
    """

    if len(tiles) != len(overlaps):
        raise ValueError(
            f"The length of tiles and overlaps must be the same. Got {len(tiles)} and {len(overlaps)}."
        )

    num_non_tiled = len(arr.shape) - len(tiles)
    orig_shape = arr.shape[num_non_tiled:]
    pad_width = ((0, 0),) * num_non_tiled + tuple(
        (overlap, overlap) for overlap in overlaps
    )

    arr = np.pad(arr, pad_width, mode=pad)
    out_arr = None

    blending = blending_map(tiles, overlaps)
    if to_device is not None:
        blending = to_device(blending)

    tiling_start = product(
        *[
            range(o, size + 2 * o, t + o)
            for size, t, o in zip(orig_shape, tiles, overlaps)
        ]
    )

    for start_indices in tqdm(tiling_start, "Applying function to tiles"):
        slicing = (...,) + tuple(
            slice(start - o, start + t + o)
            for start, t, o in zip(start_indices, tiles, overlaps)
        )

        tile = arr[slicing]
        if to_device is not None:
            tile = to_device(tile)

        out_tile = func(tile)

        if out_arr is None:
            out_arr = np.zeros(
                out_tile.shape[:num_non_tiled] + arr.shape[num_non_tiled:],
                dtype=out_dtype,
            )

        short_axes = tuple(
            i
            for i in range(len(tiles))
            if out_tile.shape[i + num_non_tiled] != blending.shape[i]
        )

        if len(short_axes) > 0:
            # ignoring right side blending on axes that the last tile doesn't overhang
            fixed_blending = blending_map(
                tiles,
                overlaps,
                ignore_right=short_axes,
            )
            if to_device is not None:
                fixed_blending = to_device(fixed_blending)
            out_tile = (
                out_tile
                * fixed_blending[
                    (...,) + tuple(slice(s) for s in out_tile.shape[num_non_tiled:])
                ]
            )
        else:
            out_tile = out_tile * blending

        out_tile = to_numpy(out_tile)

        out_arr[slicing] += out_tile

    slicing = (...,) + tuple(
        slice(o, size + o) for size, o in zip(orig_shape, overlaps)
    )
    if out_arr is None:
        raise ValueError("No tiles were processed.")

    out_arr = out_arr[slicing]

    assert (
        out_arr.shape[num_non_tiled:] == orig_shape
    ), f"{out_arr.shape} != {orig_shape}"

    return out_arr
