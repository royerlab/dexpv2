import numpy as np

from dexpv2.intensity import equalize_views, estimate_quantiles


def test_estimate_quantiles_basic() -> None:
    arr = np.array([1, 2, 3, 4, 5])
    lower, upper = estimate_quantiles(arr)
    assert lower == np.quantile(arr, 0.01)
    assert upper == np.quantile(arr, 0.9999)


def test_estimate_quantiles_with_downsampling() -> None:
    arr = np.arange(100)
    lower, upper = estimate_quantiles(arr, downsampling=10)
    assert lower == np.quantile(arr[::10], 0.01)
    assert upper == np.quantile(arr[::10], 0.9999)


def test_equalize_views_basic() -> None:
    view_0 = np.array([1, 2, 3])
    view_1 = np.array([2, 3, 4])
    equalized_view_0, equalized_view_1 = equalize_views(
        view_0.copy(), view_1.copy(), downsampling=None
    )
    # Check if the brightest view remains unchanged
    assert np.array_equal(equalized_view_1, view_1)
    # Check if the other view was adjusted correctly
    assert not np.array_equal(equalized_view_0, view_0)


def test_equalize_views_identical_views() -> None:
    view_0 = np.array([1, 2, 3])
    view_1 = np.array([1, 2, 3])
    equalized_view_0, equalized_view_1 = equalize_views(
        view_0.copy(), view_1.copy(), lower=0, upper=1, downsampling=None
    )
    # Check if views remain unchanged since they are already equal
    assert np.array_equal(equalized_view_0, view_0)
    assert np.array_equal(equalized_view_1, view_1)


def test_equalization_range() -> None:
    view_0 = np.arange(10, 101)
    view_1 = np.arange(20, 30)
    equalized_view_0, equalized_view_1 = equalize_views(
        view_0.copy(), view_1.copy(), lower=0, upper=1, downsampling=None
    )
    # Check if views remain unchanged since they are already equal
    assert np.array_equal(equalized_view_0, view_0)
    assert equalized_view_1.min() == 10
    assert equalized_view_1.max() == 100


def test_equalize_views_with_downsampling() -> None:
    view_0 = np.linspace(0, 10, 100)
    view_1 = np.linspace(0, 20, 100)
    kwargs = {"downsampling": 2}
    _, _ = equalize_views(view_0.copy(), view_1.copy(), **kwargs)
    #  it didn't crash
