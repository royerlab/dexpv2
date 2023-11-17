import itertools

import numpy as np
import pytest
from scipy.ndimage import affine_transform
from skimage import data
from skimage.metrics import structural_similarity

from dexpv2.cuda import to_numpy
from dexpv2.warp import (
    apply_warp,
    estimate_multiscale_warp,
    estimate_warp,
    filter_low_quality_vectors,
)

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("n_dim,multiscale", itertools.product([2, 3], [False, True]))
def test_warp(n_dim: int, multiscale: bool, interactive_test: bool) -> None:

    angle = 10
    angle_rad = np.deg2rad(angle)
    image = data.cells3d()[:, 1, :, :]

    if n_dim == 2:
        image = image[image.shape[0] // 2]
        affine_matrix = np.eye(2)
        offset = 5
    else:
        # small skew in 3D
        affine_matrix = np.asarray([[1, 0, 0], [0, 1, 0], [np.sin(angle_rad), 0, 1]])
        offset = 0

    # Transform image
    moved_image = affine_transform(image, affine_matrix, offset=offset)

    tile = (12, 48, 48)[-n_dim:]
    overlap = (4, 8, 8)[-n_dim:]

    if multiscale:
        warp_field = estimate_multiscale_warp(
            image,
            moved_image,
            n_scales=3,
            tile=tile,
            overlap=overlap,
            to_device=cp.asarray,
        )
    else:
        warp_field = estimate_warp(
            image, moved_image, tile=tile, overlap=overlap, to_device=cp.asarray
        )

    if n_dim == 2:
        warp_field = warp_field[:-1, ...]

    warped_image = apply_warp(cp.asarray(moved_image, dtype=np.float32), warp_field)
    warped_image = to_numpy(warped_image)

    if interactive_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(
            moved_image, colormap="yellow", blending="additive", visible=False
        )
        viewer.add_image(image, colormap="red", blending="additive")
        viewer.add_image(warped_image, colormap="blue", blending="additive")

        viewer.axes.visible = True
        viewer.dims.ndisplay = n_dim
        viewer.camera.angles = (0, 0, 0)

        napari.run()

    ssmi = structural_similarity(
        warped_image,
        image,
        data_range=image.max() - image.min(),
    )

    # Close enough
    assert ssmi > 0.85


@pytest.mark.parametrize("n_dim", [2, 3])
def test_basic_functionality(n_dim: int):
    # Test basic functionality with a simple case
    rng = cp.random.default_rng(0)
    shape = (10,) * n_dim
    warp_field = rng.random((n_dim + 1, *shape))
    warp_field[-1] = rng.uniform(0, 1, warp_field.shape[1:])
    score_threshold = 0.1
    num_iters = 5

    corrected_field = filter_low_quality_vectors(warp_field, score_threshold, num_iters)
    corrected_field = to_numpy(corrected_field)
    warp_field = to_numpy(warp_field)

    # Assert the shape and type of the output matches the input
    assert corrected_field.shape == warp_field.shape
    assert corrected_field.dtype == warp_field.dtype

    mask = warp_field[-1] <= score_threshold

    # scores remain unchanged
    np.testing.assert_allclose(warp_field[-1], corrected_field[-1])

    # all inside mask must have changed
    not_changed = np.sum(np.allclose(warp_field[:-1, mask], corrected_field[:-1, mask]))
    assert (
        not_changed <= 5
    )  # at least 5 could be not changed because of the random nature of the test

    # other values must be unchanged
    np.testing.assert_allclose(warp_field[:-1, ~mask], corrected_field[:-1, ~mask])


def test_no_correction_needed():
    # Test case where no correction is needed (all scores above threshold)
    warp_field = cp.random.rand(3, 10, 10)
    warp_field[-1, :, :] = cp.random.uniform(0.6, 1, (10, 10))  # All scores above 0.5
    score_threshold = 0.5
    num_iters = 3

    corrected_field = filter_low_quality_vectors(warp_field, score_threshold, num_iters)
    corrected_field = to_numpy(corrected_field)
    warp_field = to_numpy(warp_field)

    # Assert that the output is unchanged from the input
    assert np.array_equal(corrected_field, warp_field)
