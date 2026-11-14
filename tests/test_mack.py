"""Mack (1993) standard errors: package equality and published values."""

import chainladder as cl
import numpy as np
import pytest

from src.mack import mack
from tests.conftest import to_cl_triangle


def _reference(tri, origins):
    dev = cl.Development(sigma_interpolation="mack").fit_transform(to_cl_triangle(tri, origins))
    return cl.MackChainladder().fit(dev)


def test_mack_by_origin_matches_package(case):
    _, tri, origins, _ = case
    ref = _reference(tri, origins)
    res = mack(tri)
    np.testing.assert_allclose(res.se[1:], ref.mack_std_err_.values[0, 0, 1:, -1], rtol=1e-8)
    np.testing.assert_allclose(res.reserve, np.nan_to_num(ref.ibnr_.values[0, 0, :, -1]), rtol=1e-10)


def test_mack_total_matches_package(case):
    _, tri, origins, _ = case
    ref = _reference(tri, origins)
    res = mack(tri)
    np.testing.assert_allclose(res.total_se, ref.total_mack_std_err_.values.ravel()[0], rtol=1e-8)
    np.testing.assert_allclose(res.total_process_se, ref.total_process_risk_.values[0, 0, 0, -1], rtol=1e-8)
    np.testing.assert_allclose(res.total_parameter_se, ref.total_parameter_risk_.values[0, 0, 0, -1], rtol=1e-8)


@pytest.mark.parametrize(
    "name, reserve, total_se",
    [
        # Taylor & Ashe data as in Mack (1993), Table 2: reserve 18,680,856, s.e. 2,447,095
        ("genins", 18_680_856, 2_447_095),
        # RAA data, Mack (1994) / R ChainLadder::MackChainLadder(RAA): IBNR 52,135, s.e. 26,909
        ("raa", 52_135, 26_909),
    ],
)
def test_mack_published_values(name, reserve, total_se):
    res = mack(cl.load_sample(name).values[0, 0])
    assert res.reserve.sum() == pytest.approx(reserve, abs=1)
    assert res.total_se == pytest.approx(total_se, abs=1)


def test_mack_variance_decomposition(case):
    _, tri, _, _ = case
    res = mack(tri)
    np.testing.assert_allclose(res.se**2, res.process_se**2 + res.parameter_se**2)
    assert res.se[0] == 0.0
