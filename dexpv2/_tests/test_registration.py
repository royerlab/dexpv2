import numpy as np
from scipy.ndimage import affine_transform
from skimage import data
from skimage.metrics import structural_similarity

from dexpv2.registration import apply_affine_transform, estimate_affine_transform


def test_affine_registration(interactive_test: bool):
    # Load the example nuclei image from skimage and keep only one channel
    angle = 10
    image = data.cells3d()[:, 1, :, :]

    # Create an affine transform for skewing
    angle_rad = np.deg2rad(angle)
    affine_matrix = np.array([[1, 0, 0], [0, 1, 0], [np.sin(angle_rad), 0, 1]])
    offset = np.array([0, 0, -image.shape[-1] / 4])

    # Transform image
    moved_image = affine_transform(image, affine_matrix, offset=offset)

    # Estimate reverse transform
    transform, reg_image = estimate_affine_transform(
        image,
        moved_image,
        voxel_size=(1, 1, 1),
        reg_px_size=1,
        return_reg_moving=True,
        aff_metric="meansquares",  # this works better in this case, but not on our embryos
    )
    other_reg_image = apply_affine_transform(
        transform, moved_image, voxel_size=(1, 1, 1)
    )

    np.testing.assert_allclose(reg_image, other_reg_image)

    if interactive_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, colormap="red", blending="additive")
        viewer.add_image(moved_image, colormap="blue", blending="additive")
        viewer.add_image(reg_image, colormap="green", blending="additive")

        viewer.axes.visible = True
        viewer.dims.ndisplay = 3
        viewer.camera.angles = (0, 0, 0)

        napari.run()

    # ignoring regions that were out of the field of view
    ssmi = structural_similarity(
        reg_image[..., :195],
        image[..., :195],
        data_range=image.max() - image.min(),
    )

    # Fair enough
    assert ssmi > 0.80
