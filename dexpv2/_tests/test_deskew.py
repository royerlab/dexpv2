import numpy as np
from scipy.ndimage import affine_transform
from skimage import data
from skimage.metrics import structural_similarity

from dexpv2.deskew import deskew


def test_skew_deskew(interactive_test: bool):
    # Load the example nuclei image from skimage and keep only one channel
    angle = 60
    image = data.cells3d()[:, 1, :, :]

    # Create an affine transform for skewing
    angle_rad = np.deg2rad(90 - angle)
    affine_matrix = np.array([[1, 0, 0], [0, 1, 0], [np.cos(angle_rad), 0, 1]])

    # Skew the image
    skewed_image = affine_transform(image, affine_matrix)

    # deskew happens in a different axis
    transposed_skewed_image = np.transpose(skewed_image, (2, 1, 0))

    # using arbitrary scale because our deskewing only work with integers
    pixel_size = 2
    step_size = 1

    # Deskew the image
    deskewed_image = deskew(transposed_skewed_image, pixel_size, step_size, angle)

    # Fixing scaling
    deskewed_image = affine_transform(
        deskewed_image,
        np.linalg.inv(np.diag([step_size, step_size, pixel_size])),
    )
    del transposed_skewed_image

    offset = image.shape[-1] - deskewed_image.shape[-1]

    if interactive_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, colormap="red", blending="additive")
        viewer.add_image(skewed_image, colormap="blue", blending="additive")
        viewer.add_image(
            deskewed_image, colormap="green", blending="additive", translate=(0, 0, -4)
        )

        viewer.axes.visible = True
        viewer.dims.ndisplay = 3
        viewer.camera.angles = (0, 0, 0)

        napari.run()

    # Ignoring x because it's changed
    assert deskewed_image.shape[1:-1] == image.shape[1:-1]
    # Checking z separately because we remove one slice
    assert deskewed_image.shape[0] + 1 == image.shape[0]

    ssmi = structural_similarity(
        deskewed_image[..., 60 + 4 :],  # shift of 4 aligns the data
        image[1:, ..., 60 : -offset - 4],  # 60 removes the empty part
        data_range=image.max() - image.min(),
    )

    # Fair enough
    assert ssmi > 0.80
