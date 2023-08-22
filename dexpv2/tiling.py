from itertools import product
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from dexpv2.cuda import to_numpy


def blending_map(
    tile: Tuple[int, ...],
    overlap: Tuple[int, ...],
    ignore_right: Tuple[int, ...] = tuple(),
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Blending map for tiles with specified overlaps.
    """
    blending = 1
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
        line_blending = np.concatenate((left_border, np.ones(tile[i]), right_border))
        line_blending = line_blending[
            (None,) * i + (...,) + (None,) * (len(tile) - i - 1)
        ]
        blending = blending * line_blending

    return blending


def apply_tiled(
    arr: np.ndarray,
    func: Callable[[ArrayLike], ArrayLike],
    tile: Tuple[int, ...],
    overlap: Union[int, Tuple[int, ...]],
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
    tile : Tuple[int, ...]
        Sizes of the tiles along each dimension.
    overlap : Union[int, Tuple[int, ...]]
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

    blending = blending_map(tile, overlap)
    if to_device is not None:
        blending = to_device(blending)

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

        in_tile = arr[slicing]
        if to_device is not None:
            in_tile = to_device(in_tile)

        out_tile = func(in_tile)
        del in_tile

        if out_arr is None:
            out_arr = np.zeros(
                out_tile.shape[:num_non_tiled] + arr.shape[num_non_tiled:],
                dtype=out_dtype,
            )

        short_axes = tuple(
            i
            for i in range(len(tile))
            if out_tile.shape[i + num_non_tiled] != blending.shape[i]
        )

        if len(short_axes) > 0:
            # ignoring right side blending on axes that the last tile doesn't overhang
            fixed_blending = blending_map(
                tile,
                overlap,
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
        del out_tile

    slicing = (...,) + tuple(slice(o, size + o) for size, o in zip(orig_shape, overlap))
    if out_arr is None:
        raise ValueError("No tiles were processed.")

    out_arr = out_arr[slicing]

    assert (
        out_arr.shape[num_non_tiled:] == orig_shape
    ), f"{out_arr.shape} != {orig_shape}"

    return out_arr