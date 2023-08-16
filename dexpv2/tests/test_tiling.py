from typing import Callable, Optional, Tuple

import numpy as np
import pytest
import torch as th
from numpy.typing import ArrayLike

from dexpv2.cuda import torch_default_device
from dexpv2.tiling import apply_tiled, blending_map

try:
    import cupy as xp
except ImportError:
    import numpy as xp


@pytest.mark.parametrize(
    "tiles,overlaps", [((12, 12), (4, 4)), ((6, 6), (2, 2)), ((3, 4, 4), (1, 2, 2))]
)
def test_blending_map(
    tiles: Tuple[int, ...],
    overlaps: Tuple[int, ...],
) -> None:

    blending = blending_map(tiles, overlaps)
    expected_shape = tuple(t + 2 * o for t, o in zip(tiles, overlaps))

    assert blending.shape == expected_shape, f"{blending.shape} != {expected_shape}"


@pytest.mark.parametrize(
    "in_shape,out_shape,tiles,overlaps,to_device",
    [
        ((1, 37, 53), (2, 37, 53), (12, 12), (4, 4), None),
        ((9, 15), (9, 15), (3, 4), (1, 2), None),
        (
            (1, 23, 41, 64),
            (3, 23, 41, 64),
            (3, 4, 7),
            (1, 2, 3),
            lambda x: xp.asarray(x),
        ),
        (
            (2, 23, 17),
            (2, 23, 17),
            (6, 6),
            (2, 2),
            lambda x: th.as_tensor(x, device=torch_default_device()),
        ),
    ],
)
def test_apply_tiled(
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    tiles: Tuple[int, ...],
    overlaps: Tuple[int, ...],
    to_device: Optional[Callable[[ArrayLike], ArrayLike]],
) -> None:

    in_arr = np.ones(in_shape)

    if in_shape != out_shape:
        shape_factor = tuple(o // i for i, o in zip(in_shape, out_shape))
        expander = np.ones(shape_factor, dtype=in_arr.dtype)
        if to_device is not None:
            expander = to_device(expander)
    else:
        expander = None

    def identity(x: ArrayLike) -> ArrayLike:
        if expander is not None:
            x = expander * x
        return x

    out_arr = apply_tiled(in_arr, identity, tiles, overlaps, to_device=to_device)

    expected_out_arr = np.ones(out_shape)

    np.testing.assert_allclose(out_arr, expected_out_arr)
