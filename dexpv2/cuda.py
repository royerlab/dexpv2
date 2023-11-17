import importlib
import logging
from contextlib import contextmanager, nullcontext
from types import ModuleType
from typing import Any, Dict, Generator, Iterator, Optional, Tuple, cast

import numpy as np
import torch as th
from numpy.typing import ArrayLike

LOG = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupy.cuda.memory import malloc_managed
    from cupy.cuda.texture import TextureObject

    LOG.info("cupy found.")
except (ModuleNotFoundError, ImportError):
    LOG.info("cupy not found.")
    cp = None


CUPY_MODULES = {
    "scipy": "cupyx.scipy",
    "skimage": "cucim.skimage",
}


@contextmanager
def unified_memory() -> Generator:
    """
    Initializes cupy's unified memory allocation.
    cupy's functions will run slower but it will spill memory memory into cpu without crashing.
    """
    # starts unified memory
    if cp is not None:
        previous_allocator = cp.cuda.get_allocator()
        cp.cuda.set_allocator(malloc_managed)

    yield

    # ends unified memory
    if cp is not None:
        cp.clear_memo()
        cp.cuda.set_allocator(previous_allocator)


@contextmanager
def maybe_unified_memory(value: int, threshold: int = 5368709120) -> Generator:
    """
    Initializes cupy's unified memory allocation if it exceeds a threshold.
    cupy's functions will run slower but it will spill memory memory into cpu without crashing.

    Parameters
    ----------
    value : int
        Value to compare with threshold.
    threshold : int
        Threshold in bytes 5368709120 = 1280 * 2048 * 2048.

    Returns
    -------
    Generator
        Unified memory context.
    """
    if value > threshold:
        with unified_memory() as ctx:
            yield ctx
    else:
        with nullcontext() as ctx:
            yield ctx


def torch_default_device() -> th.device:
    """
    Returns "gpu", "mps" or "cpu" devices depending on their availability.

    Returns
    -------
    th.device
        Torch fastest device.
    """
    if th.cuda.is_available():
        device = th.cuda.device_count() - 1
    elif th.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return th.device(device)


def import_module(module: str, submodule: str) -> ModuleType:
    """Import GPU accelerated module if available, otherwise returns CPU version.

    Parameters
    ----------
    module : str
        Main python module (e.g. skimage, scipy)

    submodule : str
        Secondary python module (e.g. morphology, ndimage)

    Returns
    -------
    ModuleType
        Imported submodule.
    """
    cupy_module_name = f"{CUPY_MODULES[module]}.{submodule}"
    try:
        pkg = importlib.import_module(cupy_module_name)
        LOG.info(f"{cupy_module_name} found.")

    except (ModuleNotFoundError, ImportError):

        pkg = importlib.import_module(f"{module}.{submodule}")
        LOG.info(f"{cupy_module_name} not found. Using cpu equivalent")

    return pkg


def to_numpy(arr: ArrayLike) -> ArrayLike:
    """Moves array to cpu and converts to numpy, if it's already there nothing is done."""

    if isinstance(arr, th.Tensor):
        arr = arr.cpu().numpy()

    elif cp is not None and isinstance(arr, cp.ndarray):
        arr = arr.get()

    else:
        arr = np.asarray(arr)

    return arr


def to_texture_memory(
    array: ArrayLike,
    channel_axis: Optional[int] = None,
    normalize_values: bool = False,
    normalize_coords: bool = False,
    sampling_mode: str = "linear",
    address_mode: str = "clamp",
) -> Tuple["TextureObject", ArrayLike]:
    """
    Transfers an array to texture memory for GPU processing using CuPy.

    This function creates a CUDA texture object and a CUDA array. The texture object can be
    used for various GPU-based image processing tasks where texture memory offers performance
    benefits. This function is particularly useful for applications that require fast image
    or signal processing on the GPU.

    Parameters
    ----------
    array : ArrayLike
        The array to be transferred to texture memory. Typically, this should be a CuPy ndarray.
    channel_axis : int, optional
        Indicates the axis that contains the channels. Defaults to None.
    normalize_values : bool, optional
        Whether to normalize the values in the texture memory. Defaults to False.
    normalize_coords : bool, optional
        Whether to use normalized coordinates. Defaults to False.
    sampling_mode : str, optional
        The texture sampling mode. Either "nearest" or "linear". Defaults to "linear".
    address_mode : str, optional
        The addressing mode for out-of-bound texture coordinates. Can be "border", "clamp",
        "mirror", or "wrap". Defaults to "clamp".

    Returns
    -------
    Tuple["TextureObject", ArrayLike]
        A tuple containing the texture object and the CUDA array.

    Raises
    ------
    ValueError
        If CuPy is not found, or if invalid values are provided for dtype, address_mode, sampling_mode,
        or normalize_values.

    Examples
    --------
    >>> import cupy as cp
    >>> array = cp.array([[1, 2], [3, 4]], dtype=cp.float32)
    >>> texture_object, cuda_array = to_texture_memory(array)
    """

    if cp is None:
        raise ValueError("Cupy not found. Texture memory only available with cupy.")

    _CHANNEL_TYPE = {
        "f": cp.cuda.runtime.cudaChannelFormatKindFloat,
        "i": cp.cuda.runtime.cudaChannelFormatKindSigned,
        "u": cp.cuda.runtime.cudaChannelFormatKindUnsigned,
    }

    _ADDRESS_MODE = {
        "border": cp.cuda.runtime.cudaAddressModeBorder,
        "clamp": cp.cuda.runtime.cudaAddressModeClamp,
        "mirror": cp.cuda.runtime.cudaAddressModeMirror,
        "wrap": cp.cuda.runtime.cudaAddressModeWrap,
    }

    _TEXTURE_SAMPLING = {
        "nearest": cp.cuda.runtime.cudaFilterModePoint,
        "linear": cp.cuda.runtime.cudaFilterModeLinear,
    }

    _NORMALIZE_MODE = {
        True: cp.cuda.runtime.cudaReadModeNormalizedFloat,
        False: cp.cuda.runtime.cudaReadModeElementType,
    }

    dtype = np.dtype(array.dtype)

    if channel_axis is None:
        num_channels = 1
        spatial_shape = list(array.shape)
    else:
        num_channels = array.shape[channel_axis]
        spatial_shape = list(array.shape)
        spatial_shape.pop(channel_axis)
        array = np.moveaxis(array, channel_axis, -1)
        array = array.reshape(*spatial_shape[:-1], -1)  # (Z), Y, X * C

    if num_channels not in (1, 2, 4):
        raise ValueError(
            f"Invalid number of channels ({num_channels}). Only 1, 2, or 4 channels are supported."
        )

    nbits = 8 * dtype.itemsize
    channels = (nbits,) * num_channels + (0,) * (4 - num_channels)

    LOG.info(f"Creating channel format descriptor: {channels} {dtype.kind[0]}")

    try:
        format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(
            *channels, _CHANNEL_TYPE[dtype.kind[0]]
        )
    except KeyError:
        raise ValueError(
            f"Invalid dtype ({dtype}). Valid must start with: {list(_CHANNEL_TYPE.keys())}"
        )

    cuda_array = cp.cuda.texture.CUDAarray(format_descriptor, *spatial_shape[::-1])

    LOG.info("Creating resource descriptor ...")

    resource_descriptor = cp.cuda.texture.ResourceDescriptor(
        cp.cuda.runtime.cudaResourceTypeArray, cuArr=cuda_array
    )

    try:
        LOG.info("Creating texture descriptor ...")
        texture_descriptor = cp.cuda.texture.TextureDescriptor(
            addressModes=(_ADDRESS_MODE[address_mode],) * array.ndim,
            filterMode=_TEXTURE_SAMPLING[sampling_mode],
            readMode=_NORMALIZE_MODE[normalize_values],
            sRGB=None,
            borderColors=None,
            normalizedCoords=normalize_coords,
            maxAnisotropy=None,
        )
    except KeyError:
        for k, v in cast(
            Iterator[Tuple[Any, Dict]],
            zip(
                (address_mode, sampling_mode, normalize_values),
                (_ADDRESS_MODE, _TEXTURE_SAMPLING, _NORMALIZE_MODE),
            ),
        ):
            if k not in v:
                raise ValueError(
                    f"Invalid value for {k}. Valid values are: {list(v.keys())}"
                )

    LOG.info("Creating texture object ...")
    texture_object = cp.cuda.texture.TextureObject(
        resource_descriptor, texture_descriptor
    )

    LOG.info("Synchronize ...")
    # required by previous dexp code, otherwise it would fail
    cp.cuda.runtime.deviceSynchronize()

    LOG.info("Copying array content ...")
    LOG.info(f"CUDAArray: {(cuda_array.height, cuda_array.width, cuda_array.depth)}")
    LOG.info(f"Array shape: {array.shape}")
    cuda_array.copy_from(cp.ascontiguousarray(cp.asarray(array)))

    del format_descriptor, texture_descriptor, resource_descriptor

    LOG.info("Texture memory setup is done.")

    return texture_object, cuda_array
