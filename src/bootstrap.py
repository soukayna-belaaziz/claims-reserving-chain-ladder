"""Over-dispersed Poisson (ODP) bootstrap of chain-ladder reserves.

Reference: P. England & R. Verrall (1999, 2002), "Stochastic claims reserving in
general insurance", British Actuarial Journal 8(3); and M. Shapland (2016), CAS
Monograph No. 4 "Using the ODP Bootstrap Model: A Practitioner's Guide".

Algorithm (vectorised over simulations):
  1. Fit the chain ladder; back-fit expected cumulative values from the latest diagonal
     and difference them into expected incremental values m_ij.
  2. Unscaled Pearson residuals r_ij = (X_ij - m_ij) / sqrt(m_ij); scale parameter
     phi = sum r^2 / (n - p) with n observed cells and p = 2 * n_origin - 1 parameters.
  3. Resample residuals (optionally bias-adjusted by sqrt(n / (n - p))), build pseudo
     incremental triangles X* = m + r* sqrt(m), re-estimate the development factors
     and project the future incremental means (parameter error).
  4. Add process error by drawing each future incremental from a Gamma distribution
     with mean m* and variance phi * m* (the ODP mean-variance relationship).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deterministic import volume_weighted_ldf
from .triangle import is_observed, latest_diagonal, latest_lag, to_incremental


@dataclass(frozen=True)
class OdpFit:
    """Fitted values and residuals of the ODP / chain-ladder model."""

    expected_incremental: np.ndarray
    unscaled_residuals: np.ndarray
    scale_phi: float
    n_obs: int
    n_params: int


@dataclass(frozen=True)
class BootstrapResult:
    """Simulated reserves (without tail) by accident year (shape: n_sims x n_origin)."""

    reserves_by_origin: np.ndarray
    fit: OdpFit

    @property
    def total(self) -> np.ndarray:
        """Simulated total reserve."""
        return self.reserves_by_origin.sum(axis=1)


def odp_fit(cum: np.ndarray) -> OdpFit:
    """Chain-ladder expected incremental values and unscaled Pearson residuals."""
    n_orig, n_dev = cum.shape
    ldf = volume_weighted_ldf(cum)
    obs = is_observed(cum)
    expected_cum = np.full(cum.shape, np.nan)
    lag = latest_lag(cum)
    latest = latest_diagonal(cum)
    for i in range(n_orig):
        expected_cum[i, lag[i]] = latest[i]
        for j in range(lag[i] - 1, -1, -1):
            expected_cum[i, j] = expected_cum[i, j + 1] / ldf[j]
    m = to_incremental(expected_cum)
    x = to_incremental(cum)
    resid = np.where(obs, (x - m) / np.sqrt(np.abs(m)), np.nan)
    n_obs = int(obs.sum())
    n_params = n_orig + n_dev - 1
    phi = float(np.nansum(resid**2) / (n_obs - n_params))
    return OdpFit(m, resid, phi, n_obs, n_params)


def _batch_ldf(cum: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Volume-weighted LDFs for a batch of triangles, shape (n_sims, n_dev - 1)."""
    num = np.where(obs[None, :, 1:], cum[:, :, 1:], 0.0).sum(axis=1)
    den = np.where(obs[None, :, 1:], cum[:, :, :-1], 0.0).sum(axis=1)
    return num / den


def _pseudo_development(
    cum: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
    bias_adjust: bool,
    drop_zero_residuals: bool,
    center_residuals: bool,
) -> tuple[OdpFit, np.ndarray, np.ndarray]:
    """Resample Pearson residuals into pseudo triangles; return the fit, pseudo cumulative data and their LDFs."""
    fit = odp_fit(cum)
    obs = is_observed(cum)
    n_orig, n_dev = cum.shape
    m = np.nan_to_num(fit.expected_incremental)

    pool = fit.unscaled_residuals[obs]
    if drop_zero_residuals:
        pool = pool[~np.isclose(pool, 0.0, atol=1e-10)]
    if bias_adjust:
        pool = pool * np.sqrt(fit.n_obs / (fit.n_obs - fit.n_params))
    if center_residuals:
        pool = pool - pool.mean()

    draws = rng.choice(pool, size=(n_sims, n_orig, n_dev), replace=True)
    pseudo_inc = np.where(obs[None], m[None] + draws * np.sqrt(np.abs(m))[None], 0.0)
    pseudo_cum = np.cumsum(pseudo_inc, axis=2)
    return fit, pseudo_cum, _batch_ldf(pseudo_cum, obs)


def _gamma_process(mean: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """ODP process error: Gamma draw with mean |m| and variance phi |m|, sign of m restored."""
    shape = np.abs(mean) / phi
    sim = np.where(shape > 0, rng.gamma(np.maximum(shape, 1e-12), phi), 0.0)
    return np.sign(mean) * sim


def odp_bootstrap(
    cum: np.ndarray,
    n_sims: int = 10_000,
    seed: int = 42,
    bias_adjust: bool = True,
    process_error: bool = True,
    drop_zero_residuals: bool = True,
    center_residuals: bool = False,
) -> BootstrapResult:
    """Simulate the predictive distribution of the chain-ladder reserve.

    Parameters
    ----------
    cum: cumulative triangle (NaN below the latest diagonal)
    bias_adjust: multiply residuals by sqrt(n / (n - p)) (England & Verrall 2002)
    process_error: add Gamma process error; False gives the parameter-error-only distribution
    drop_zero_residuals: remove the structurally zero residuals (corner cells) before resampling
    center_residuals: subtract the mean of the residual pool (Shapland 2016)
    """
    rng = np.random.default_rng(seed)
    fit, pseudo_cum, ldf = _pseudo_development(cum, n_sims, rng, bias_adjust, drop_zero_residuals, center_residuals)
    obs = is_observed(cum)
    n_orig, n_dev = cum.shape

    lag = latest_lag(cum)
    future_inc = np.zeros((n_sims, n_orig, n_dev))
    for i in range(n_orig):
        level = pseudo_cum[:, i, lag[i]]
        for j in range(lag[i] + 1, n_dev):
            new_level = level * ldf[:, j - 1]
            future_inc[:, i, j] = new_level - level
            level = new_level

    if process_error:
        future_mask = ~obs
        future_inc[:, future_mask] = _gamma_process(future_inc[:, future_mask], fit.scale_phi, rng)

    return BootstrapResult(future_inc.sum(axis=2), fit)


@dataclass(frozen=True)
class OneYearResult:
    """Simulated one-year claims development result by accident year (n_sims x n_origin).

    ``CDR = U^I - U^{I+1}``: ultimate at the valuation date minus the re-estimated ultimate
    one year later. Positive values are run-off gains, negative values losses.
    """

    cdr_by_origin: np.ndarray
    ultimate_now: np.ndarray
    fit: OdpFit

    @property
    def total(self) -> np.ndarray:
        """Simulated total CDR."""
        return self.cdr_by_origin.sum(axis=1)


def odp_one_year_cdr(
    cum: np.ndarray,
    n_sims: int = 10_000,
    seed: int = 42,
    bias_adjust: bool = True,
    drop_zero_residuals: bool = True,
) -> OneYearResult:
    """Re-reserving ("actuary-in-the-box") bootstrap of the one-year CDR under the ODP model.

    For every simulation (Ohlsson & Lauzeningks 2009; R ChainLadder ``CDR.BootChainLadder``):

    1. draw pseudo development factors f* from the residual bootstrap (parameter error);
    2. simulate the next calendar diagonal from the *actual* latest diagonal:
       ``X_{i,l+1} ~ Gamma(mean = C_{i,l} (f*_l - 1), var = phi * mean)`` (process error);
    3. append it to the real triangle, re-estimate all factors with the chain ladder and
       re-project the ultimate: ``U^{I+1}``;
    4. CDR = ``U^I - U^{I+1}``, with ``U^I`` the chain-ladder ultimate on the real data.
    """
    rng = np.random.default_rng(seed)
    fit, _, ldf_star = _pseudo_development(cum, n_sims, rng, bias_adjust, drop_zero_residuals, False)
    n_orig, n_dev = cum.shape
    obs = is_observed(cum)
    lag = latest_lag(cum)
    latest = latest_diagonal(cum)
    f0 = volume_weighted_ldf(cum)
    ultimate_now = latest * np.array([np.prod(f0[lag[i]:]) for i in range(n_orig)])

    open_rows = np.flatnonzero(lag < n_dev - 1)
    mean_next = latest[open_rows][None] * (ldf_star[:, lag[open_rows]] - 1.0)
    next_inc = _gamma_process(mean_next, fit.scale_phi, rng)

    ext = np.broadcast_to(np.nan_to_num(cum), (n_sims, n_orig, n_dev)).copy()
    ext[:, open_rows, lag[open_rows] + 1] = latest[open_rows][None] + next_inc
    obs_ext = obs.copy()
    obs_ext[open_rows, lag[open_rows] + 1] = True
    ldf_next = _batch_ldf(ext, obs_ext)

    ultimate_next = np.repeat(latest[None], n_sims, axis=0)
    for i in open_rows:
        l1 = lag[i] + 1
        ultimate_next[:, i] = ext[:, i, l1] * np.prod(ldf_next[:, l1:], axis=1)
    return OneYearResult(ultimate_now[None] - ultimate_next, ultimate_now, fit)


def summarise(samples: np.ndarray, percentiles: tuple[float, ...] = (50, 75, 95, 99.5)) -> dict[str, float]:
    """Mean, standard deviation, coefficient of variation and percentiles of a simulated reserve."""
    out = {"mean": float(samples.mean()), "std": float(samples.std(ddof=1))}
    out["cv"] = out["std"] / out["mean"]
    for p in percentiles:
        out[f"p{p:g}"] = float(np.percentile(samples, p))
    return out
