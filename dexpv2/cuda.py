import logging

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
