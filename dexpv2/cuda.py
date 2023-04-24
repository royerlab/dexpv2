import logging

import torch as th

LOG = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupy.cuda.memory import malloc_managed

    LOG.info("cupy found.")
except (ModuleNotFoundError, ImportError):
    LOG.info("cupy not found.")
    cp = None


def setup_unified_memory() -> None:
    """
    Initializes cupy's unified memory allocation.
    cupy's functions will run slower but it will spill memory memory into cpu without crashing.
    """
    if cp is None:
        return
    cp.cuda.set_allocator(malloc_managed)


def torch_default_device() -> th.device:
    """
    Returns "gpu", "mps" or "cpu" devices depending on their availability.

    Returns
    -------
    th.device
        Torch fastest device.
    """
    if th.cuda.is_available():
        device = "cuda"
    elif th.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return th.device(device)
