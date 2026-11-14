"""Trend-adjusted a-priori loss ratios: package equality and limiting cases."""

import chainladder as cl
import numpy as np
import pytest

from src.deterministic import apriori_elr_from_mature_years, cape_cod, chain_ladder
from src.priors import generalised_cape_cod, loss_ratio_trend, trended_mature_prior
from tests.conftest import premium_triangle, to_cl_triangle


@pytest.mark.parametrize("trend, decay", [(0.0, 1.0), (-0.045, 1.0), (0.03, 0.75), (0.05, 0.0)])
def test_generalised_cape_cod_matches_package(premium_case, trend, decay):
    _, tri, origins, prem = premium_case
    res = chain_ladder(tri)
    ult, elr = generalised_cape_cod(res.latest, res.cdf, prem, trend, decay)
    ref = cl.CapeCod(trend=trend, decay=decay).fit(to_cl_triangle(tri, origins), sample_weight=premium_triangle(prem, origins))
    np.testing.assert_allclose(ult, ref.ultimate_.values[0, 0, :, -1], rtol=1e-10)
    np.testing.assert_allclose(elr, ref.detrended_apriori_.values[0, 0, :, -1], rtol=1e-10)


def test_untrended_cape_cod_is_the_classical_one(premium_case):
    _, tri, _, prem = premium_case
    res = chain_ladder(tri)
    ult, elr = generalised_cape_cod(res.latest, res.cdf, prem)
    ref_ult, ref_elr = cape_cod(res.latest, res.cdf, prem)
    np.testing.assert_allclose(ult, ref_ult)
    np.testing.assert_allclose(elr, ref_elr)


def test_weight_mask_removes_a_year_from_the_elr(premium_case):
    _, tri, _, prem = premium_case
    res = chain_ladder(tri)
    mask = np.ones(len(prem), bool)
    mask[3] = False
    _, elr = generalised_cape_cod(res.latest, res.cdf, prem, weight_mask=mask)
    keep = mask
    expected = res.latest[keep].sum() / (prem[keep] / res.cdf[keep]).sum()
    np.testing.assert_allclose(elr, expected)


def test_trended_prior_reduces_to_mature_average():
    ult = np.array([80.0, 90.0, 70.0, 60.0])
    prem = np.array([100.0, 100.0, 100.0, 100.0])
    np.testing.assert_allclose(trended_mature_prior(ult, prem, 0.0, n_mature=2), apriori_elr_from_mature_years(ult, prem, 2))
    # with a trend, year 3 gets the mature years' loss ratios brought forward 3 and 2 years
    expected = (80.0 * 1.1**3 + 90.0 * 1.1**2) / 200.0
    assert trended_mature_prior(ult, prem, 0.1, n_mature=2)[3] == pytest.approx(expected)


def test_loss_ratio_trend_recovers_exact_trend():
    origins = np.arange(2000, 2010)
    prem = np.full(10, 100.0)
    ult = prem * 0.7 * 0.96 ** (origins - 2009) * (1 + 0.001 * np.sin(origins))  # tiny noise for a finite s.e.
    fit = loss_ratio_trend(ult, prem, np.linspace(1.0, 2.0, 10), origins)
    assert fit.annual_trend == pytest.approx(-0.04, abs=2e-4)
    assert np.exp(fit.intercept) == pytest.approx(0.7, rel=1e-3)
    assert abs(fit.t_stat) > 100
    assert loss_ratio_trend(ult, prem, np.ones(10), origins, exclude=(2003,)).n_years == 9
