from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from dexpv2.crosscorr import phase_cross_corr
from dexpv2.cuda import to_texture_memory
from dexpv2.tiling import apply_tiled_stacked
from dexpv2.utils import translation_slicing

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
        raise ValueError(
            "CUDA Texture does not support length 3 channels."
            "To avoid this remove the score channel (last) as `apply_warp(warped_image[:-1], ....)"
        )

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
    arr = arr / arr.std()
    return arr


def estimate_warp(
    fixed: ArrayLike,
    moving: ArrayLike,
    tile: Tuple[int, ...],
    overlap: Union[int, Tuple[int, ...]],
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
    return_warped: bool = False,
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

    def _correlate(*args: ArrayLike) -> np.ndarray:
        moving, fixed = args
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
        return np.asarray((*shift, score.item()), dtype=np.float32)

    warp_field = apply_tiled_stacked(
        fixed,
        moving,
        func=_correlate,
        tile=tile,
        overlap=overlap,
        to_device=to_device,
    )

    return warp_field
