import pytest

from dexpv2.cuda import to_texture_memory

cp = pytest.importorskip("cupy")


def test_to_texture_memory_valid_input() -> None:
    array = cp.zeros((64, 64), dtype=cp.float32)
    texture_object, cuda_array = to_texture_memory(array)
    assert isinstance(texture_object, cp.cuda.texture.TextureObject)
    assert (cuda_array.height, cuda_array.width) == array.shape
    del texture_object, cuda_array


def test_to_texture_memory_invalid_address_mode() -> None:
    array = cp.zeros((64, 64), dtype=cp.float32)
    with pytest.raises(ValueError):
        to_texture_memory(array, address_mode="invalid_mode")
