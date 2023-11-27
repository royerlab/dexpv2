import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.constants import DEXPV2_DEBUG
from dexpv2.crosscorr import phase_cross_corr
from dexpv2.cuda import import_module, to_numpy, to_texture_memory
from dexpv2.tiling import BlendingMap, apply_tiled_stacked
from dexpv2.utils import translation_slicing

LOG = logging.getLogger(__name__)


_WARP_KERNELS = {
    2: r"""
extern "C"{
    __global__ void warp_2d(float* warped_image,
                            cudaTextureObject_t input_image,
                            cudaTextureObject_t vector_field,
                            int width,
                            int height)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            // coordinates in coord-normalised vector_field texture:
            float u = float(x)/width;
            float v = float(y)/height;

            // Obtain linearly interpolated vector at (u,v):
            float2 vector = tex2D<float2>(vector_field, u, v);

            // Obtain the shifted coordinates of the source voxel:
            // flip axis order to match numpy order
            float sx = 0.5f + float(x) - vector.y;
            float sy = 0.5f + float(y) - vector.x;

            // Sample source image for voxel value:
            float value = tex2D<float>(input_image, sx, sy);

            //printf("(%f, %f)=%f\n", sx, sy, value);

            // Store interpolated value:
            warped_image[y * width + x] = value;

            // TODO: supersampling would help in regions for which warping misses voxels in
            // the source image.
            // improve: adaptive supersampling would automatically use the vector field divergence to
            // determine where to super sample and by how much.
        }
    }
}""",
    3: r"""
extern "C"{
    __global__ void warp_3d(float* warped_image,
                            cudaTextureObject_t input_image,
                            cudaTextureObject_t vector_field,
                            int width,
                            int height,
                            int depth)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < width && y < height && z < depth)
        {
            // coordinates in coord-normalised vector_field texture:
            float u = float(x)/width;
            float v = float(y)/height;
            float w = float(z)/depth;
            //printf("(%f,%f,%f)\n", u, v, w);

            // Obtain linearly interpolated vector at (u,v,w):
            float4 vector = tex3D<float4>(vector_field, u, v, w);

            //printf("(%f,%f,%f,%f)\n", vector.x, vector.y, vector.z, vector.w);

            // Obtain the shifted coordinates of the source voxel,
            // flip axis order to match numpy order:
            float sx = 0.5f + float(x) - vector.z;
            float sy = 0.5f + float(y) - vector.y;
            float sz = 0.5f + float(z) - vector.x;

            // Sample source image for voxel value:
            float value = tex3D<float>(input_image, sx, sy, sz);

            //printf("(%f, %f, %f)=%f\n", sx, sy, sz, value);

            // Store interpolated value:
            warped_image[z*width*height + y*width + x] = value;

            //TODO: supersampling would help in regions for which warping misses voxels in the source image,
            //better: adaptive supersampling would automatically use the vector field
            // divergence to determine where to super sample and by how much.
        }
    }
}""",
}


def apply_warp(
    image: ArrayLike,
    warp_field: ArrayLike,
    cuda_block_size: Optional[int] = None,
) -> ArrayLike:
    """
    Apply a warp field to an image using CUDA Texture, requires cupy.

    This function warps an input 2D or 3D image based on the provided warp field.

    Parameters
    ----------
    image : ArrayLike
        The input image to be warped. Can be a 2D or 3D array.
    warp_field : ArrayLike
        The warp field array. Its dimensions should be one more than the
        input image (i.e., 3D for a 2D image and 4D for a 3D image).
        The warp field specifies the displacement at each pixel/voxel.
    cuda_block_size : Optional[int], default None
        The size of the CUDA block for GPU computation. If None, a default
        size of 8 is used. Only relevant when using CUDA.

    Raises
    ------
    ValueError
        If the input image is not 2D or 3D, or if the dimensions of the
        warp field do not match the expected dimensions based on the image.

    Returns
    -------
    ArrayLike
        The warped image, having the same shape and type as the input image.

    Notes
    -----
    The function utilizes CuPy's RawKernel feature to execute a custom CUDA kernel,
    allowing for efficient image warping on the GPU. It also handles the conversion
    of the image and warp field to texture memory for optimized performance on CUDA.

    Examples
    --------
    >>> image = np.random.rand(100, 100)
    >>> warp_field = np.random.rand(100, 100, 2)
    >>> warped_image = apply_warp(image, warp_field)

    See Also
    --------
    to_texture_memory : Function used for converting arrays to texture memory.
    cupy.RawKernel : CuPy feature used for executing custom CUDA kernels.
    """

    if cuda_block_size is None:
        cuda_block_size = 8 if image.ndim == 3 else 8

    import cupy as cp

    if image.ndim != 2 and image.ndim != 3:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D instead!")

    if warp_field.ndim != image.ndim + 1:
        raise ValueError(
            f"Warp field must be {image.ndim + 1}D, got {warp_field.ndim}D instead!"
        )

    warp_kernel = cp.RawKernel(_WARP_KERNELS[image.ndim], f"warp_{image.ndim}d")
    warped_image = cp.empty_like(image)

    shape = image.shape[::-1]
    grid = tuple((s + cuda_block_size - 1) // cuda_block_size for s in shape)

    image_text, _ = to_texture_memory(
        image,
        normalize_coords=False,
        sampling_mode="linear",
    )

    if image.ndim == 2 and warp_field.shape[0] == 3:
        # IMPORTANT: removing scores to avoid 3 channels in texture memory, 3 is not allowed.
        warp_field = warp_field[:-1]

    field_text, _ = to_texture_memory(
        warp_field,
        channel_axis=0,
        normalize_coords=True,
        sampling_mode="linear",
        address_mode="clamp",
    )

    warp_kernel(
        grid,
        (cuda_block_size,) * image.ndim,
        (warped_image, image_text, field_text, *shape),
    )

    return warped_image


def _standardize(arr: ArrayLike) -> ArrayLike:
    arr = arr - arr.mean()
    arr = arr / (np.linalg.norm(arr) - 1e-8)
    return arr


def estimate_warp(
    fixed: ArrayLike,
    moving: ArrayLike,
    tile: Tuple[int, ...],
    overlap: Union[int, Tuple[int, ...]],
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
    **kwargs,
) -> np.ndarray:
    """
    Estimate the warp field that aligns a 'moving' image to a 'fixed' image.

    This function applies a tiled correlation approach to estimate the warp field
    needed to align two images. It divides the images into tiles, computes the
    phase correlation for each tile, and then calculates a score indicating the
    quality of alignment.

    Parameters
    ----------
    fixed : ArrayLike
        The reference image to which 'moving' is aligned.
    moving : ArrayLike
        The image that needs to be warped to align with 'fixed'.
    tile : Tuple[int, ...]
        The size of the tiles into which the images are divided for processing.
    overlap : Union[int, Tuple[int, ...]]
        The amount of overlap between adjacent tiles, either as a single integer
        or a tuple specifying the overlap for each dimension.
    to_device : Callable[[ArrayLike], ArrayLike], optional
        A function that transfers data to the device where computation will occur,
        by default identity function (no transfer).
    **kwargs
        Additional keyword arguments passed to the phase correlation function.

    Returns
    -------
    np.ndarray
        An array representing the warp field. Each element contains the shift (in
        (z), y, x) and the correlation score for the corresponding tile.

    Notes
    -----
    The function internally defines `_correlate`, a helper function to compute the
    phase correlation between corresponding tiles of the 'fixed' and 'moving' images.

    The warp field is computed by applying `_correlate` in a tiled and stacked manner
    using the `apply_tiled_stacked` function.

    See Also
    --------
    phase_cross_corr : Function used for computing phase correlation.
    apply_tiled_stacked : Function for applying operations in a tiled and stacked manner.

    Examples
    --------
    >>> fixed = np.random.rand(100, 100)
    >>> moving = np.random.rand(100, 100)
    >>> tile = (50, 50)
    >>> overlap = 10
    >>> warp_field = estimate_warp(fixed, moving, tile, overlap)
    """

    if isinstance(overlap, int):
        overlap = (overlap,) * len(tile)

    blending = BlendingMap(tile, overlap, num_non_tiled=0, to_device=to_device)

    def _correlate(*args: ArrayLike) -> np.ndarray:
        moving, fixed = args

        fixed = blending(fixed)
        moving = blending(moving)

        if np.all(fixed < 1e-8):
            return np.zeros(fixed.ndim + 1, dtype=np.float32)

        shift = np.asarray(
            phase_cross_corr(
                fixed,
                moving,
                to_device=to_device,
                **kwargs,
            )
        )
        fixed_slice = fixed[translation_slicing(-shift)].ravel()
        moving_slice = moving[translation_slicing(shift)].ravel()
        score = np.dot(_standardize(fixed_slice), _standardize(moving_slice))
        output = np.asarray((*shift, score.item()), dtype=np.float32)
        LOG.info(f"Tiled warp vector: {output}")
        print(output)
        return output

    warp_field = apply_tiled_stacked(
        fixed,
        moving,
        func=_correlate,
        tile=tile,
        overlap=overlap,
        to_device=to_device,
    )

    return warp_field


def filter_low_quality_vectors(
    warp_field: ArrayLike,
    score_threshold: float,
    num_iters: int,
) -> ArrayLike:
    """
    Correct low-quality vectors in a warp field based on a score threshold.

    This function iteratively smooths parts of the warp field that have
    a quality score below a specified threshold. The correction is performed
    using a uniform filter from SciPy's ndimage module.

    Parameters
    ----------
    warp_field : ArrayLike
        The warp field to be corrected. The last channel of this field
        should represent the quality score of each vector.
    score_threshold : float
        The threshold below which vectors are considered low quality and
        subject to correction.
    num_iters : int
        The number of iterations to perform the smoothing operation.

    Returns
    -------
    ArrayLike
        The corrected warp field. The shape and type of the output
        will match the input warp field.

    Notes
    -----
    The correction process involves applying a uniform filter to smooth
    the warp vectors that are below the quality threshold. This smoothing
    is performed iteratively for the specified number of iterations.

    This function can be particularly useful in post-processing warp fields
    obtained from image registration algorithms, especially in cases where
    some vectors in the field are unreliable or noisy.

    Examples
    --------
    >>> warp_field = np.random.rand(100, 100, 3)
    >>> warp_field[:, :, -1] = np.random.uniform(0, 1, (100, 100))  # Last channel as score
    >>> corrected_field = correct_low_quality_vector(warp_field, 0.5, 5)

    See Also
    --------
    scipy.ndimage.uniform_filter : Function used for applying the smoothing filter.
    """
    if warp_field.ndim != warp_field.shape[0]:
        raise ValueError(
            "Warp field first dimension must have length=D+1, where D+1 is the dimension of"
            f"the image, plus the last score-axis. Got shape {warp_field.shape} instead."
        )

    mask = warp_field[-1] < score_threshold
    if not np.any(mask):
        return warp_field

    if np.all(mask):
        LOG.warning("All vectors are low quality, returning zero warp field.")
        warp_field[:-1, ...] = 0
        return warp_field

    ndi = import_module("scipy", "ndimage")

    for i in range(warp_field.shape[0] - 1):
        warp_field[i][mask] = 0
        for _ in range(num_iters):
            corrected_field = ndi.uniform_filter(warp_field[i], size=3)
            warp_field[i][mask] = corrected_field[mask]

    return warp_field


def estimate_multiscale_warp(
    fixed: ArrayLike,
    moving: ArrayLike,
    n_scales: int,
    tile: Tuple[int, ...],
    overlap: Union[int, Tuple[int, ...]],
    score_threshold: float = 0.5,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
    **kwargs,
) -> np.ndarray:
    """
    Estimate the warp field that aligns a 'moving' image to a 'fixed' image.

    This function applies a tiled correlation approach to estimate the warp field
    needed to align two images. It divides the images into tiles, computes the
    phase correlation for each tile, and then calculates a score indicating the
    quality of alignment.

    Parameters
    ----------
    fixed : ArrayLike
        The reference image to which 'moving' is aligned.
    moving : ArrayLike
        The image that needs to be warped to align with 'fixed'.
    n_scales : int
        The number of scales at which to compute the warp field.
    tile : Tuple[int, ...]
        The size of the tiles into which the images are divided for processing.
    overlap : Union[int, Tuple[int, ...]]
        The amount of overlap between adjacent tiles, either as a single integer
        or a tuple specifying the overlap for each dimension.
    score_threshold : float, optional
        The threshold below which vectors are considered low quality and
        subject to correction.
    to_device : Callable[[ArrayLike], ArrayLike], optional
        A function that transfers data to the device where computation will occur,
        by default identity function (no transfer).
    **kwargs
        Additional keyword arguments passed to the phase correlation function.

    Returns
    -------
    np.ndarray
        An array representing the warp field. Each element contains the shift (in
        (z), y, x) and the correlation score for the corresponding tile.
    """
    ndi = import_module("scipy", "ndimage")
    transform = import_module("skimage", "transform")

    fixed = to_device(fixed)
    moving = to_device(moving.astype(np.float32))
    warp_field = None

    for i in range(n_scales):

        LOG.info(f"Estimating warp field at scale {i+1}/{n_scales}")

        if warp_field is None:
            warped_moving = moving
        else:
            warped_moving = apply_warp(moving, warp_field)

            if DEXPV2_DEBUG:
                import napari

                viewer = napari.Viewer()
                viewer.add_image(
                    to_numpy(fixed), name="fixed", colormap="red", blending="additive"
                )
                viewer.add_image(
                    to_numpy(warped_moving),
                    name="warped moving",
                    colormap="green",
                    blending="additive",
                )
                viewer.add_image(
                    to_numpy(moving),
                    name="moving",
                    colormap="blue",
                    blending="additive",
                    visible=False,
                )
                # print(moving.shape, warp_field.shape)
                # viewer.add_image(
                #     to_numpy(warp_field[-1]),
                #     scale=np.asarray(moving.shape) / warp_field.shape[1:],
                #     name="Warp score",
                #     colormap="magma",
                #     blending="additive",
                #     contrast_limits=(0, 1),
                #     visible=False,
                # )
                print(f"Scale {i+1}/{n_scales}")
                napari.run()

        downsampling_factor = 0.5 ** (n_scales - i - 1)

        if downsampling_factor < 1:
            scaled_moving = ndi.zoom(warped_moving, downsampling_factor, order=1)
            scaled_fixed = ndi.zoom(fixed, downsampling_factor, order=1)
        else:
            scaled_moving = warped_moving
            scaled_fixed = fixed

        new_warp_field = to_device(
            estimate_warp(
                scaled_fixed,
                scaled_moving,
                tile,
                overlap,
                to_device=to_device,
                **kwargs,
            )
        )
        # scaling to original interpolation space
        new_warp_field[:-1] /= downsampling_factor

        new_warp_field = filter_low_quality_vectors(
            new_warp_field, score_threshold=score_threshold, num_iters=5
        )

        print(downsampling_factor)
        print(new_warp_field[:-1].min(), new_warp_field[:-1].max())

        if warp_field is not None:
            for c in range(warp_field.shape[0] - 1):
                new_warp_field[c] += transform.resize(
                    warp_field[c],
                    new_warp_field.shape[1:],
                    order=1,
                    anti_aliasing=False,
                    clip=True,
                )

        warp_field = new_warp_field

    return warp_field
