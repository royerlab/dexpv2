import numpy as np
import pytest

from dexpv2.crop import find_moving_bboxes_consensus, to_slice


def test_find_moving_bboxes_consensus_normal() -> None:
    bboxes = np.array([[0, 0, 10, 10], [2, 2, 12, 12]])
    shifts = np.array([[1, 1], [1, 1]])
    expected_output = np.array([[1, 1, 11, 11], [2, 2, 12, 12]])
    output = find_moving_bboxes_consensus(bboxes, shifts)
    np.testing.assert_array_equal(output, expected_output)


def test_find_moving_bboxes_consensus_mismatch_error() -> None:
    bboxes = np.array([[0, 0, 10, 10]])
    shifts = np.array([[1, 1], [1, 1]])
    with pytest.raises(ValueError):
        find_moving_bboxes_consensus(bboxes, shifts)


def test_find_moving_bboxes_consensus_dimension_error() -> None:
    bboxes = np.array([[0, 0, 10]])
    shifts = np.array([[1, 1]])
    with pytest.raises(ValueError):
        find_moving_bboxes_consensus(bboxes, shifts)


def test_to_slice() -> None:
    bbox = np.array([0, 0, 10, 10])
    expected_output = (slice(0, 10), slice(0, 10))
    output = to_slice(bbox)
    assert output == expected_output
