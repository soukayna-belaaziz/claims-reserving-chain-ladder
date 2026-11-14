"""Diagnostics: package equality for Mack's tests and exactness of the back-test."""

import numpy as np
import pytest

from src.data import upper_triangle
from src.diagnostics import (
    calendar_year_test,
    development_correlation_test,
    diagonal_backtest,
    mack_standardised_residuals,
)
from tests.conftest import to_cl_triangle


def test_calendar_year_test_matches_package(case):
    _, tri, origins, _ = case
    ref = to_cl_triangle(tri, origins).valuation_correlation(p_critical=0.1, total=True)
    res = calendar_year_test(tri)
    assert res.z == pytest.approx(ref.z.values.ravel()[0])
    assert res.expected == pytest.approx(ref.z_expectation.values.ravel()[0])
    assert res.variance == pytest.approx(ref.z_variance.values.ravel()[0])


def test_development_correlation_matches_package(case):
    _, tri, origins, _ = case
    ref = to_cl_triangle(tri, origins).development_correlation(p_critical=0.5)
    res = development_correlation_test(tri)
    assert res.t == pytest.approx(ref.t_expectation.values.ravel()[0])
    np.testing.assert_allclose(res.ci, ref.confidence_interval)


def test_backtest_is_exact_on_a_deterministic_chain_ladder_triangle():
    """If every row develops with the same factors, actual = expected on the held-out diagonal."""
    f = np.array([2.0, 1.5, 1.2, 1.1, 1.05])
    first = np.array([100.0, 120.0, 90.0, 110.0, 130.0, 105.0])
    square = first[:, None] * np.concatenate([[1.0], np.cumprod(f)])[None, :]
    res = diagonal_backtest(upper_triangle(square))
    np.testing.assert_allclose(res.ave[1:], 0.0, atol=1e-9)
    assert np.isnan(res.ave[0])  # oldest year: no factor estimable without the diagonal


def test_standardised_residuals_shape(case):
    _, tri, _, _ = case
    r = mack_standardised_residuals(tri)
    assert r.shape == (tri.shape[0], tri.shape[1] - 1)
    assert np.isfinite(r[~np.isnan(r)]).all()
