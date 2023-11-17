import numpy as np
import pytest
from scipy.ndimage import affine_transform
from skimage import data
from skimage.metrics import structural_similarity

from dexpv2.cuda import to_numpy
from dexpv2.warp import apply_warp, estimate_warp

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("n_dim", [2, 3])
def test_warp(n_dim: int, interactive_test: bool) -> None:

    angle = 10
    angle_rad = np.deg2rad(angle)
    image = data.cells3d()[:, 1, :, :]

    if n_dim == 2:
        image = image[image.shape[0] // 2]
        affine_matrix = np.eye(2)
        offset = 10
    else:
        # small skew in 3D
        affine_matrix = np.asarray([[1, 0, 0], [0, 1, 0], [np.sin(angle_rad), 0, 1]])
        offset = 0

    # Transform image
    moved_image = affine_transform(image, affine_matrix, offset=offset)

    warp_field = estimate_warp(
        image, moved_image, tile=(16, 64, 64)[-n_dim:], overlap=4, to_device=cp.asarray
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
