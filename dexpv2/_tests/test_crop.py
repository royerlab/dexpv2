import numpy as np
import pytest

from dexpv2.crop import find_moving_bboxes_consensus, fix_crop_slice, to_slice


def test_find_moving_bboxes_consensus_normal() -> None:
    bboxes = np.array([[0, 0, 10, 10], [2, 2, 12, 12]])
    shifts = np.array([[1, 1], [1, 1]])
    expected_output = np.array(
        [[0, 0, 10, 10, 0, 0, 10, 10], [2, 2, 12, 12, 1, 1, 11, 11]]
    )
    output = find_moving_bboxes_consensus(bboxes, shifts, shape=(14, 14))
    np.testing.assert_array_equal(output, expected_output)


def test_find_moving_bboxes_consensus_mismatch_error() -> None:
    bboxes = np.array([[0, 0, 10, 10]])
    shifts = np.array([[1, 1], [1, 1]])
    with pytest.raises(ValueError):
        find_moving_bboxes_consensus(bboxes, shifts, shape=(12, 12))


def test_find_moving_bboxes_consensus_dimension_error() -> None:
    bboxes = np.array([[0, 0, 10]])
    shifts = np.array([[1, 1]])
    with pytest.raises(ValueError):
        find_moving_bboxes_consensus(bboxes, shifts, shape=(12, 12))


def test_to_slice() -> None:
    bbox = np.array([0, 0, 10, 10])
    expected_output = (slice(0, 10), slice(0, 10))
    output = to_slice(bbox)
    assert output == expected_output


def test_identity_fix_crop_slice() -> None:
    crop_slice = (slice(0, 8), slice(3, 10))
    shape = (10, 10)

    src_slice, dst_slice = fix_crop_slice(crop_slice, shape)
    exp_dst_slice = (slice(0, 8), slice(0, 7))

    assert crop_slice == src_slice
    assert dst_slice == exp_dst_slice


def test_fix_crop_slice() -> None:
    crop_slice = (slice(-2, 12), slice(-3, 5), slice(2, 11))
    shape = (10, 10, 10)

    src_slice, dst_slice = fix_crop_slice(crop_slice, shape)

    exp_src_slice = (slice(0, 10), slice(0, 5), slice(2, 10))
    exp_dst_slice = (slice(0, 10), slice(0, 5), slice(0, 8))

    assert src_slice == exp_src_slice
    assert dst_slice == exp_dst_slice
