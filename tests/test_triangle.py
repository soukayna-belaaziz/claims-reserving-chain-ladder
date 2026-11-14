import numpy as np

from src.data import upper_triangle
from src.triangle import latest_diagonal, latest_lag, remove_last_diagonal, to_cumulative, to_incremental


def test_incremental_roundtrip(case):
    _, tri, _, _ = case
    back = to_cumulative(to_incremental(tri))
    np.testing.assert_allclose(back[~np.isnan(tri)], tri[~np.isnan(tri)])


def test_latest_diagonal_and_lag():
    sq = np.arange(1, 17, dtype=float).reshape(4, 4)
    tri = upper_triangle(sq)
    np.testing.assert_array_equal(latest_lag(tri), [3, 2, 1, 0])
    np.testing.assert_array_equal(latest_diagonal(tri), [4, 7, 10, 13])


def test_remove_last_diagonal():
    tri = upper_triangle(np.ones((4, 4)))
    reduced = remove_last_diagonal(tri)
    assert np.isnan(reduced).sum() == np.isnan(tri).sum() + 4
    np.testing.assert_array_equal(latest_lag(reduced[:-1]), [2, 1, 0])
