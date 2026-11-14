"""Diagnostics of the chain-ladder assumptions.

* Mack standardised residuals of the individual development factors
* Mack (1994) test for calendar-year (diagonal) effects
* Mack (1994) test for correlation between successive development factors
* Back-test: hold out the latest calendar diagonal and compare actual vs expected

Reference: T. Mack (1994), "Measuring the variability of chain ladder reserve
estimates", CAS Forum Spring 1994, Appendices G and H.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.stats import norm, rankdata

from .deterministic import volume_weighted_ldf
from .mack import mack_sigma2
from .triangle import calendar_index, is_observed, latest_diagonal, link_ratios, remove_last_diagonal


def mack_standardised_residuals(cum: np.ndarray) -> np.ndarray:
    """``(F_ij - f_j) * sqrt(C_ij) / sigma_j``; under Mack's model they have mean 0, variance ~1."""
    ldf = volume_weighted_ldf(cum)
    sigma = np.sqrt(mack_sigma2(cum, ldf))
    factors = link_ratios(cum)
    return (factors - ldf[None, :]) * np.sqrt(cum[:, :-1]) / sigma[None, :]


@dataclass(frozen=True)
class CalendarYearTest:
    """Result of Mack's calendar-year effect test."""

    z_by_diagonal: np.ndarray
    s_by_diagonal: np.ndarray
    l_by_diagonal: np.ndarray
    z: float
    expected: float
    variance: float
    ci: tuple[float, float]
    reject: bool


def _z_moments(n: int) -> tuple[float, float]:
    """Mean and variance of Z = min(S, L) for n factors under the null of no diagonal effect."""
    if n == 0:
        return 0.0, 0.0
    m = (n - 1) // 2
    c = comb(n - 1, m)
    ez = n / 2 - c * n / 2**n
    varz = n * (n - 1) / 4 - c * n * (n - 1) / 2**n + ez - ez**2
    return ez, varz


def calendar_year_test(cum: np.ndarray, confidence: float = 0.95) -> CalendarYearTest:
    """Mack (1994) test: are small/large development factors clustered on some diagonals?

    In each development column the factors are classified as *small* (below the column
    median) or *large* (above); the median itself is discarded. For each calendar diagonal
    Z_j = min(S_j, L_j); under independence sum Z_j is approximately normal with the
    moments of :func:`_z_moments`. Too *small* a Z means one diagonal is dominated by
    small or large factors (e.g. a change in claims handling or inflation shock).
    """
    factors = link_ratios(cum)
    small = np.zeros(factors.shape, dtype=bool)
    large = np.zeros(factors.shape, dtype=bool)
    for j in range(factors.shape[1]):
        obs = is_observed(factors[:, j])
        if obs.sum() == 0:
            continue
        ranks = rankdata(factors[obs, j])
        med = np.median(ranks)
        small[np.where(obs)[0][ranks < med], j] = True
        large[np.where(obs)[0][ranks > med], j] = True
    # calendar index of a factor F_ij is the calendar period of its numerator, i + j + 1
    cal = calendar_index(factors.shape) + 1
    diags = np.arange(1, cum.shape[0])
    s = np.array([small[cal == d].sum() for d in diags])
    lg = np.array([large[cal == d].sum() for d in diags])
    z = np.minimum(s, lg)
    moments = np.array([_z_moments(int(n)) for n in s + lg])
    ez, varz = moments[:, 0].sum(), moments[:, 1].sum()
    q = norm.ppf(0.5 + confidence / 2)
    ci = (ez - q * np.sqrt(varz), ez + q * np.sqrt(varz))
    return CalendarYearTest(z, s, lg, float(z.sum()), float(ez), float(varz), ci, not (ci[0] <= z.sum() <= ci[1]))


@dataclass(frozen=True)
class DevelopmentCorrelationTest:
    """Result of Mack's test for correlation of adjacent development factors."""

    t_k: np.ndarray
    weights: np.ndarray
    t: float
    variance: float
    ci: tuple[float, float]
    reject: bool


def development_correlation_test(cum: np.ndarray, confidence: float = 0.5) -> DevelopmentCorrelationTest:
    """Mack (1994) weighted Spearman test of correlation between columns j and j+1 of factors.

    For every pair of adjacent columns only accident years observed in both are used.
    T = sum_k (m_k - 1) T_k / sum_k (m_k - 1), with Var(T) = 1 / sum_k (m_k - 1).
    Mack recommends a 50% confidence band because the test is only approximate.
    """
    factors = link_ratios(cum)
    t_k, w = [], []
    for j in range(factors.shape[1] - 1):
        both = is_observed(factors[:, j]) & is_observed(factors[:, j + 1])
        m = int(both.sum())
        if m < 2:
            continue
        r = rankdata(factors[both, j + 1])
        s = rankdata(factors[both, j])
        t_k.append(1 - 6 * np.sum((r - s) ** 2) / (m**3 - m))
        w.append(m - 1)
    t_k_arr, w_arr = np.array(t_k), np.array(w, dtype=float)
    t = float(np.sum(w_arr * t_k_arr) / w_arr.sum())
    var = float(1.0 / w_arr.sum())
    q = norm.ppf(0.5 + confidence / 2)
    ci = (-q * np.sqrt(var), q * np.sqrt(var))
    return DevelopmentCorrelationTest(t_k_arr, w_arr, t, var, ci, not (ci[0] <= t <= ci[1]))


@dataclass(frozen=True)
class DiagonalBacktest:
    """Actual vs expected emergence on a held-out calendar diagonal."""

    actual_incremental: np.ndarray
    expected_incremental: np.ndarray
    std_error: np.ndarray  # Mack one-period process s.e. sqrt(sigma^2 C)

    @property
    def ave(self) -> np.ndarray:
        """Actual minus expected incremental emergence by accident year."""
        return self.actual_incremental - self.expected_incremental

    @property
    def z_score(self) -> np.ndarray:
        """Standardised AvE (actual - expected) / s.e."""
        return self.ave / self.std_error


def diagonal_backtest(cum: np.ndarray) -> DiagonalBacktest:
    """Remove the latest diagonal, refit, predict it one period ahead and compare with actual.

    The newest accident year has no prior observation and is excluded (its first value is
    not a development of anything). Expected value: f_j * C_ij; one-period variance from
    Mack's assumption (2): sigma_j^2 * C_ij, with f and sigma estimated without the diagonal.
    """
    reduced = remove_last_diagonal(cum)
    ldf = volume_weighted_ldf(reduced[:-1, :-1])
    sigma2 = mack_sigma2(reduced[:-1, :-1], ldf)
    prev = latest_diagonal(reduced[:-1])
    actual = latest_diagonal(cum[:-1])
    n = cum.shape[0]
    j = (n - 2) - np.arange(n - 1)  # lag of the previous cell for accident years 0..n-2
    # accident year 0 sits at the last lag in the held-out triangle: no factor beyond it
    valid = j < len(ldf)
    jj = np.clip(j, 0, len(ldf) - 1)
    exp_cum = np.where(valid, prev * ldf[jj], np.nan)
    se = np.where(valid, np.sqrt(sigma2[jj] * prev), np.nan)
    return DiagonalBacktest(actual - prev, exp_cum - prev, se)
