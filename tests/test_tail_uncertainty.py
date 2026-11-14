"""Inverse power tail vs chainladder, candidate tails and the tail-randomised simulation."""

import chainladder as cl
import numpy as np
import pytest

from src.deterministic import inverse_power_tail, volume_weighted_ldf
from src.tail_uncertainty import lognormal_tail_sigma, paid_tail_candidates, simulate_tails, unpaid_with_tail
from tests.conftest import to_cl_triangle


def test_inverse_power_tail_matches_package(case):
    _, tri, origins, _ = case
    ldf = volume_weighted_ldf(tri)
    if (ldf[2:] > 1.00001).sum() < 2:
        return
    dev = cl.Development().fit_transform(to_cl_triangle(tri, origins))
    ref = cl.TailCurve(curve="inverse_power", fit_period=(36, None)).fit(dev).tail_.values[0, 0]
    np.testing.assert_allclose(inverse_power_tail(ldf, start_lag=2)[0], ref, rtol=1e-8)


def test_candidates_contain_base_case(premium_case):
    from src.data import load_segments, upper_triangle
    from src.pipeline import select_tails

    name = premium_case[0].split("_")[0]
    seg = load_segments()[name]
    tp, ti = upper_triangle(seg.paid), upper_triangle(seg.incurred)
    cand = paid_tail_candidates(tp, ti)
    assert cand["none"] == 1.0
    assert cand["inc_to_paid_x_inc_exponential"] == pytest.approx(select_tails(tp, ti).paid)
    assert all(t >= 1.0 for t in cand.values())


def test_simulated_tails_keep_the_mean_and_span_candidates():
    cand = {"none": 1.0, "a": 1.02, "b": 1.05, "c": 1.10}
    sigma = lognormal_tail_sigma(cand)
    tails = simulate_tails(1.05, sigma, 200_000, seed=1)
    assert tails.min() > 1.0
    assert tails.mean() == pytest.approx(1.05, rel=1e-3)
    lo, hi = np.percentile(tails - 1.0, [5, 95])
    assert hi / lo == pytest.approx(0.10 / 0.02, rel=0.02)  # 90% interval ratio = candidate spread


def test_unpaid_with_constant_tail():
    latest = np.array([100.0, 50.0])
    res = np.array([[0.0, 50.0], [0.0, 30.0]])
    np.testing.assert_allclose(unpaid_with_tail(latest, res, 1.1), [10.0 + 60.0, 10.0 + 38.0])
    np.testing.assert_allclose(unpaid_with_tail(latest, res, np.array([1.0, 1.2])), [50.0, 20.0 + 46.0])
