"""From-scratch deterministic methods must reproduce the chainladder package."""

import chainladder as cl
import numpy as np

from src.deterministic import (
    bornhuetter_ferguson,
    cape_cod,
    chain_ladder,
    exponential_tail,
    volume_weighted_ldf,
)
from tests.conftest import premium_triangle, to_cl_triangle


def test_volume_weighted_ldf_matches_package(case):
    _, tri, origins, _ = case
    ref = cl.Development().fit(to_cl_triangle(tri, origins)).ldf_.values[0, 0, 0]
    np.testing.assert_allclose(volume_weighted_ldf(tri), ref, rtol=1e-10)


def test_chain_ladder_ultimate_matches_package(case):
    _, tri, origins, _ = case
    ref = cl.Chainladder().fit(to_cl_triangle(tri, origins)).ultimate_.values[0, 0, :, -1]
    np.testing.assert_allclose(chain_ladder(tri).ultimate, ref, rtol=1e-10)


def test_exponential_tail_matches_package(case):
    _, tri, origins, _ = case
    ldf = volume_weighted_ldf(tri)
    if (ldf[2:] > 1.00001).sum() < 2:
        return  # not enough factors > 1 to fit a curve
    dev = cl.Development().fit_transform(to_cl_triangle(tri, origins))
    ref = cl.TailCurve(curve="exponential", fit_period=(36, None)).fit(dev).tail_.values[0, 0]
    tail, _, _ = exponential_tail(ldf, start_lag=2)
    np.testing.assert_allclose(tail, ref, rtol=1e-8)


def test_chain_ladder_with_tail_matches_package(case):
    _, tri, origins, _ = case
    dev = cl.TailConstant(tail=1.05).fit_transform(cl.Development().fit_transform(to_cl_triangle(tri, origins)))
    ref = cl.Chainladder().fit(dev).ultimate_.values[0, 0, :, -1]
    np.testing.assert_allclose(chain_ladder(tri, tail=1.05).ultimate, ref, rtol=1e-10)


def test_bornhuetter_ferguson_matches_package(premium_case):
    _, tri, origins, prem = premium_case
    res = chain_ladder(tri)
    elr = 0.65
    mine = bornhuetter_ferguson(res.latest, res.cdf, prem, elr)
    ref = cl.BornhuetterFerguson(apriori=elr).fit(
        to_cl_triangle(tri, origins), sample_weight=premium_triangle(prem, origins)
    ).ultimate_.values[0, 0, :, -1]
    np.testing.assert_allclose(mine, ref, rtol=1e-8)


def test_cape_cod_matches_package(premium_case):
    _, tri, origins, prem = premium_case
    res = chain_ladder(tri)
    mine, elr = cape_cod(res.latest, res.cdf, prem)
    model = cl.CapeCod(trend=0.0, decay=1.0).fit(to_cl_triangle(tri, origins), sample_weight=premium_triangle(prem, origins))
    np.testing.assert_allclose(mine, model.ultimate_.values[0, 0, :, -1], rtol=1e-8)
    np.testing.assert_allclose(elr, model.apriori_.values[0, 0, 0, 0], rtol=1e-8)


def test_bf_equals_chain_ladder_when_prior_is_cl_loss_ratio(premium_case):
    """Sanity identity: with ELR_i = CL ultimate_i / premium_i, BF reproduces the chain ladder."""
    _, tri, _, prem = premium_case
    res = chain_ladder(tri)
    np.testing.assert_allclose(bornhuetter_ferguson(res.latest, res.cdf, prem, res.ultimate / prem), res.ultimate)
