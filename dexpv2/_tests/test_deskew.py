import numpy as np
from scipy.ndimage import affine_transform
from skimage import data
from skimage.metrics import structural_similarity

from dexpv2.deskew import deskew, deskewing_shift


def test_skew_deskew(interactive_test: bool):
    # Load the example nuclei image from skimage and keep only one channel
    angle = 45
    image = data.cells3d()[:, 1, :, :]
    image_shape = image.shape

    # Create an affine transform for skewing
    angle_rad = np.deg2rad(angle)
    affine_matrix = np.array([[1, 0, 0], [0, 1, 0], [np.cos(angle_rad), 0, 1]])

    # Skew the image
    skewed_image = affine_transform(image, affine_matrix)

    # Deskew the image
    shift = deskewing_shift(angle, 1, 1)
    deskewed_image = deskew(skewed_image, shift)

    offset = deskewed_image.shape[-1] - image_shape[-1]

    if interactive_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, colormap="red", blending="additive")
        viewer.add_image(deskewed_image, colormap="green", blending="additive")

        napari.run()

    # Ignoring x because it's changed
    assert deskewed_image.shape[:-1] == image_shape[:-1]

    ssmi = structural_similarity(
        deskewed_image[..., :-offset],
        image,
        data_range=image.max() - image.min(),
    )

    # Fair enough
    assert ssmi > 0.80
