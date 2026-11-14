"""ODP bootstrap: structure of the fit and agreement with chainladder.BootstrapODPSample."""

import chainladder as cl
import numpy as np
import pytest

from src.bootstrap import odp_bootstrap, odp_fit, summarise
from src.deterministic import chain_ladder
from src.triangle import latest_diagonal

GENINS = cl.load_sample("genins")
TRI = GENINS.values[0, 0]


def test_fitted_values_reproduce_chain_ladder_reserve():
    """Back-fitted ODP expectations: row sums equal the latest diagonal (ODP / chain-ladder equivalence)."""
    fit = odp_fit(TRI)
    obs = ~np.isnan(TRI)
    np.testing.assert_allclose(np.where(obs, fit.expected_incremental, 0).sum(axis=1), latest_diagonal(TRI))
    assert fit.n_params == 19 and fit.n_obs == 55
    # residuals in the two corner cells are exactly zero
    assert abs(fit.unscaled_residuals[0, -1]) < 1e-8 and abs(fit.unscaled_residuals[-1, 0]) < 1e-8


def test_seed_reproducibility():
    a = odp_bootstrap(TRI, n_sims=500, seed=3).total
    b = odp_bootstrap(TRI, n_sims=500, seed=3).total
    np.testing.assert_array_equal(a, b)


def test_parameter_error_matches_package():
    """Same settings as chainladder (no bias adjustment, centred residuals, no process error)."""
    n = 5000
    mine = odp_bootstrap(TRI, n_sims=n, seed=11, bias_adjust=False, process_error=False, center_residuals=True).total
    samples = cl.BootstrapODPSample(n_sims=n, random_state=11, hat_adj=False).fit(GENINS).resampled_triangles_
    ref = cl.Chainladder().fit(samples).ibnr_.sum("origin").values.ravel()
    assert mine.mean() == pytest.approx(ref.mean(), rel=0.01)
    assert mine.std() == pytest.approx(ref.std(), rel=0.04)


def test_full_predictive_distribution_matches_package():
    """With Gamma process error (chainladder adds it inside Chainladder.fit after fit_transform)."""
    n = 5000
    mine = odp_bootstrap(TRI, n_sims=n, seed=7, bias_adjust=False, process_error=True, center_residuals=True).total
    sims = cl.BootstrapODPSample(n_sims=n, random_state=7, hat_adj=False).fit_transform(GENINS)
    ref = cl.Chainladder().fit(sims).ibnr_.sum("origin").values.ravel()
    assert mine.mean() == pytest.approx(ref.mean(), rel=0.01)
    assert mine.std() == pytest.approx(ref.std(), rel=0.04)
    assert np.percentile(mine, 99.5) == pytest.approx(np.percentile(ref, 99.5), rel=0.04)


def test_bootstrap_mean_close_to_chain_ladder():
    res = odp_bootstrap(TRI, n_sims=5000, seed=1)
    assert res.total.mean() == pytest.approx(chain_ladder(TRI).reserve.sum(), rel=0.02)


def test_summarise_keys():
    s = summarise(np.arange(100.0))
    assert {"mean", "std", "cv", "p75", "p99.5"} <= set(s)


def test_one_year_cdr_is_centred_and_close_to_merz_wuthrich():
    """Re-reserving bootstrap: E[CDR] ~ 0 and s.d. of the same order as Merz-Wüthrich (ODP vs Mack model)."""
    from src.bootstrap import odp_one_year_cdr
    from src.merz_wuthrich import merz_wuthrich

    res = odp_one_year_cdr(TRI, n_sims=5000, seed=5)
    reserve = chain_ladder(TRI).reserve.sum()
    assert abs(res.total.mean()) < 0.01 * reserve
    assert res.total.std() == pytest.approx(merz_wuthrich(TRI).cdr_total_se, rel=0.10)
    # one-year uncertainty is smaller than the full run-off uncertainty of the same model
    assert res.total.std() < odp_bootstrap(TRI, n_sims=5000, seed=5).total.std()
    # the oldest year is fully developed: no CDR
    np.testing.assert_array_equal(res.cdr_by_origin[:, 0], 0.0)


def test_one_year_cdr_without_noise_reproduces_chain_ladder():
    """On an exact chain-ladder triangle (zero residuals, phi = 0) the CDR is identically zero."""
    from src.bootstrap import odp_one_year_cdr

    f = np.array([2.0, 1.5, 1.2, 1.1])
    first = np.array([100.0, 120.0, 90.0, 110.0, 130.0])
    sq = first[:, None] * np.concatenate([[1.0], np.cumprod(f)])[None]
    tri = np.where(np.add.outer(np.arange(5), np.arange(5)) <= 4, sq, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = odp_one_year_cdr(tri, n_sims=10, seed=0, drop_zero_residuals=False)
    np.testing.assert_allclose(res.cdr_by_origin, 0.0, atol=1e-8)
