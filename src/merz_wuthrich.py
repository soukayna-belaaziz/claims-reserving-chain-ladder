"""Merz-Wüthrich (2008) uncertainty of the one-year claims development result (CDR).

References
----------
M. Merz, M. V. Wüthrich (2008), "Modelling the claims development result for solvency
purposes", CAS E-Forum, Fall 2008, 542-568.
M. Merz, M. V. Wüthrich (2014), "Claims run-off uncertainty: the full picture",
Swiss Finance Institute Research Paper 14-69 (linearised formulas).

Setting (same assumptions as Mack 1993, square triangle with I = J):

* at time I the actuary sets ``R^I`` with the chain ladder on the data ``D_I``;
* during the next calendar year one new diagonal is observed and the actuary
  re-estimates all development factors on ``D_{I+1}`` (pure chain ladder, no tail);
* the observable claims development result of accident year i is
  ``CDR_i = R_i^I - (X_{i,I-i+1} + R_i^{I+1})``, i.e. ultimate at I minus ultimate at I+1.

``E[CDR_i | D_I] = 0`` and Merz & Wüthrich estimate ``msep(CDR_i | D_I)`` around 0:

    msep_i     = U_i^2 (Gamma_i + Delta_i)
    msep_total = sum_i msep_i + 2 sum_{i<k} U_i U_k (Upsilon_i + Lambda_i)

with, for ``l = I - i`` and ``w_j = C_{I-j,j} / S_j^{I+1}``,

    Delta_i   = s_l^2/(f_l^2 S_l^I) + sum_{j>l} w_j^2 s_j^2/(f_j^2 S_j^I)
    Gamma_i   = (1 + s_l^2/(f_l^2 C_{i,l})) prod_{j>l} (1 + w_j^2 s_j^2/(f_j^2 C_{I-j,j})) - 1
    Upsilon_i = (1 + s_l^2/(f_l^2 S_l^{I+1})) prod_{j>l} (1 + w_j^2 s_j^2/(f_j^2 C_{I-j,j})) - 1
    Lambda_i  = C_{i,l}/S_l^{I+1} s_l^2/(f_l^2 S_l^I) + sum_{j>l} w_j^2 s_j^2/(f_j^2 S_j^I)

``S_j^I`` sums the accident years used for ``f_j`` at time I, ``S_j^{I+1}`` adds the
diagonal cell ``C_{I-j,j}``. The linear approximation replaces ``prod(1+x) - 1`` by
``sum(x)``; this is what the R package ChainLadder (``CDR.MackChainLadder``) reports.

Compared with Mack's ultimate msep, the one-year view keeps the full process and
parameter error of the *next* development period only; later periods enter with the
weight ``w_j^2 <= 1`` (only the part of the estimation error of ``f_j`` that is revealed
by the next diagonal). Hence ``msep_CDR <= msep_Mack``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deterministic import volume_weighted_ldf
from .mack import mack, mack_sigma2
from .triangle import is_observed, latest_diagonal


@dataclass(frozen=True)
class MerzWuthrichResult:
    """One-year CDR standard errors next to Mack's ultimate standard errors (no tail)."""

    ultimate: np.ndarray
    reserve: np.ndarray
    cdr_se: np.ndarray  # sqrt(msep) of CDR by accident year
    cdr_total_se: float
    mack_se: np.ndarray  # ultimate (run-off) view, Mack 1993
    mack_total_se: float

    @property
    def one_year_share(self) -> float:
        """Ratio of the one-year to the ultimate standard error of the total reserve."""
        return self.cdr_total_se / self.mack_total_se


def _check_square_triangle(cum: np.ndarray) -> None:
    n_orig, n_dev = cum.shape
    i, j = np.indices(cum.shape)
    expected = i + j <= n_dev - 1
    if n_orig != n_dev or not np.array_equal(is_observed(cum), expected):
        raise ValueError("Merz-Wüthrich needs a complete upper triangle with as many origins as development periods")


def merz_wuthrich(cum: np.ndarray, linear_approximation: bool = False) -> MerzWuthrichResult:
    """Merz-Wüthrich (2008) msep of the one-year CDR by accident year and in total.

    Parameters
    ----------
    cum: cumulative upper triangle (NaN below the latest diagonal), I = J.
    linear_approximation: use ``sum(x)`` instead of ``prod(1 + x) - 1`` (Merz-Wüthrich 2014,
        R ChainLadder). The difference is in the 4th significant digit in practice.
    """
    _check_square_triangle(cum)
    n = cum.shape[0]
    big_i = n - 1  # last calendar period (0-based); J = I
    f = volume_weighted_ldf(cum)
    q = mack_sigma2(cum, f) / f**2  # sigma_j^2 / f_j^2
    diag = np.array([cum[big_i - j, j] for j in range(n)])  # C_{I-j,j}
    s_now = np.array([cum[: big_i - j, j].sum() for j in range(n - 1)])  # S_j^I
    s_next = s_now + diag[: n - 1]  # S_j^{I+1}
    latest = latest_diagonal(cum)
    ult = latest * np.array([np.prod(f[big_i - i :]) for i in range(n)])

    gamma, delta, upsilon, lam = (np.zeros(n) for _ in range(4))
    for i in range(1, n):
        lag = big_i - i
        later = np.arange(lag + 1, n - 1)
        w2 = (diag[later] / s_next[later]) ** 2
        param_later = np.sum(w2 * q[later] / s_now[later])
        proc_later = w2 * q[later] / diag[later]
        delta[i] = q[lag] / s_now[lag] + param_later
        lam[i] = cum[i, lag] / s_next[lag] * q[lag] / s_now[lag] + param_later
        if linear_approximation:
            gamma[i] = q[lag] / cum[i, lag] + proc_later.sum()
            upsilon[i] = q[lag] / s_next[lag] + proc_later.sum()
        else:
            prod_later = np.prod(1.0 + proc_later)
            gamma[i] = (1.0 + q[lag] / cum[i, lag]) * prod_later - 1.0
            upsilon[i] = (1.0 + q[lag] / s_next[lag]) * prod_later - 1.0

    msep = ult**2 * (gamma + delta)
    cross = sum(ult[i] * ult[i + 1 :].sum() * (upsilon[i] + lam[i]) for i in range(1, n))
    m = mack(cum)
    return MerzWuthrichResult(
        ultimate=ult,
        reserve=ult - latest,
        cdr_se=np.sqrt(msep),
        cdr_total_se=float(np.sqrt(msep.sum() + 2.0 * cross)),
        mack_se=m.se,
        mack_total_se=m.total_se,
    )
