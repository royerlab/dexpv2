import numpy as np
from scipy.ndimage import affine_transform
from skimage import data
from skimage.metrics import structural_similarity

from dexpv2.cuda import to_numpy
from dexpv2.registration import apply_affine_transform, estimate_affine_transform

try:
    import cupy as xp

except ImportError:
    import numpy as xp


def test_affine_registration(interactive_test: bool):
    # Load the example nuclei image from skimage and keep only one channel
    angle = 10
    image = data.cells3d()[:, 1, :, :]
    voxel_size = (0.6, 0.5, 0.4)  # just for the test

    # Create an affine transform for skewing
    angle_rad = np.deg2rad(angle)
    affine_matrix = np.asarray([[1, 0, 0], [0, 1, 0], [np.sin(angle_rad), 0, 1]])
    offset = np.asarray([0, 0, -image.shape[-1] / 4])

    # Transform image
    moved_image = affine_transform(image, affine_matrix, offset=offset)

    # Estimate reverse transform
    transform, reg_image = estimate_affine_transform(
        image,
        moved_image,
        voxel_size=voxel_size,
        reg_px_size=1,
        return_reg_moving=True,
        aff_smoothing_sigmas=(12.0, 6.0, 3.0, 0),
    )
    other_reg_image = to_numpy(
        apply_affine_transform(
            transform,
            xp.asarray(moved_image),
            voxel_size=voxel_size,
        )
    )

    if interactive_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(
            moved_image, colormap="yellow", blending="additive", visible=False
        )
        viewer.add_image(image, colormap="red", blending="additive")
        viewer.add_image(reg_image, colormap="blue", blending="additive")
        viewer.add_image(
            other_reg_image,
            colormap="green",
            blending="additive",
            name="scipy affine image",
        )

        viewer.axes.visible = True
        viewer.dims.ndisplay = 3
        viewer.camera.angles = (0, 0, 0)

        napari.run()

    ssmi = structural_similarity(
        reg_image,
        other_reg_image,
        data_range=reg_image.max() - reg_image.min(),
    )
    assert ssmi > 0.95

    # ignoring regions that were out of the field of view
    ssmi = structural_similarity(
        reg_image[..., :195],
        image[..., :195],
        data_range=image.max() - image.min(),
    )

    # Fair enough
    assert ssmi > 0.80
