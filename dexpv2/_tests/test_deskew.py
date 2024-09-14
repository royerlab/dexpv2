import numpy as np
from scipy.ndimage import affine_transform
from skimage import data
from skimage.metrics import structural_similarity

from dexpv2.deskew import deskew


def test_skew_deskew(interactive_test: bool):
    # Load the example nuclei image from skimage and keep only one channel
    angle = 45
    image = data.cells3d()[:, 1, :, :]

    # Create an affine transform for skewing
    angle_rad = np.deg2rad(angle)
    affine_matrix = np.array([[1, 0, 0], [0, 1, 0], [np.cos(angle_rad), 0, 1]])

    # Skew the image
    skewed_image = affine_transform(image, affine_matrix)

    # deskew happens in a different axis
    transposed_skewed_image = np.transpose(skewed_image, (2, 1, 0))

    # Deskew the image
    deskewed_image = deskew(transposed_skewed_image, 2, 1, angle)
    del transposed_skewed_image

    offset = deskewed_image.shape[-1] - image.shape[-1]

    if interactive_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, colormap="red", blending="additive")
        viewer.add_image(skewed_image, colormap="blue", blending="additive")
        viewer.add_image(deskewed_image, colormap="green", blending="additive")

        viewer.axes.visible = True
        viewer.dims.ndisplay = 3
        viewer.camera.angles = (0, 0, 0)

        napari.run()

    # Ignoring x because it's changed
    assert deskewed_image.shape[:-1] == image.shape[:-1]

    ssmi = structural_similarity(
        deskewed_image[..., :-offset],
        image,
        data_range=image.max() - image.min(),
    )

    # Fair enough
    assert ssmi > 0.80
