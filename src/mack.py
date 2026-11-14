"""Mack (1993) distribution-free standard error of chain-ladder reserves.

Reference: T. Mack (1993), "Distribution-free calculation of the standard error of
chain ladder reserve estimates", ASTIN Bulletin 23(2), 213-225.

Model assumptions (per accident year i, development lag k):
  (1) E[C_{i,k+1} | C_i0..C_ik] = f_k C_ik
  (2) Var[C_{i,k+1} | C_i0..C_ik] = sigma_k^2 C_ik
  (3) accident years are independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deterministic import chain_ladder
from .triangle import is_observed, latest_lag, link_ratios


@dataclass(frozen=True)
class MackResult:
    """Mack standard errors (no tail) by accident year and in total."""

    ldf: np.ndarray
    sigma2: np.ndarray
    latest: np.ndarray
    ultimate: np.ndarray
    reserve: np.ndarray
    process_se: np.ndarray
    parameter_se: np.ndarray
    se: np.ndarray
    total_se: float
    total_process_se: float
    total_parameter_se: float

    @property
    def cv(self) -> np.ndarray:
        """Coefficient of variation of the reserve by accident year."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.se / self.reserve


def mack_sigma2(cum: np.ndarray, ldf: np.ndarray) -> np.ndarray:
    """Estimate sigma_k^2; the last one is extrapolated with Mack's (1993) rule.

    ``sigma2_k = 1/(m_k - 1) * sum_i C_ik (C_{i,k+1}/C_ik - f_k)^2`` where ``m_k`` is the
    number of observed factors at lag k. For the last factor (single observation)
    Mack proposes ``min(s_{n-3}^4 / s_{n-4}^2, s_{n-4}^2, s_{n-3}^2)``.
    """
    factors = link_ratios(cum)
    n_f = len(ldf)
    sigma2 = np.full(n_f, np.nan)
    for k in range(n_f):
        obs = is_observed(factors[:, k])
        m = obs.sum()
        if m > 1:
            sigma2[k] = (cum[obs, k] * (factors[obs, k] - ldf[k]) ** 2).sum() / (m - 1)
    last = n_f - 1
    if np.isnan(sigma2[last]):
        a, b = sigma2[last - 2], sigma2[last - 1]
        sigma2[last] = min(b**2 / a, a, b) if a > 0 else 0.0
    return sigma2


def mack(cum: np.ndarray) -> MackResult:
    """Mack (1993) mean squared error of chain-ladder reserves (development within the triangle)."""
    n = cum.shape[0]
    cl = chain_ladder(cum, tail=1.0)
    ldf, proj = cl.ldf, cl.projected
    sigma2 = mack_sigma2(cum, ldf)
    # S_k: column sums of the accident years used to estimate f_k
    s_k = np.array([cum[: n - 1 - k, k].sum() for k in range(n - 1)])
    ult = proj[:, -1]
    lag = latest_lag(cum)
    ratio = sigma2 / ldf**2

    proc = np.zeros(n)
    param = np.zeros(n)
    for i in range(n):
        ks = np.arange(lag[i], n - 1)
        proc[i] = ult[i] ** 2 * np.sum(ratio[ks] / proj[i, ks])
        param[i] = ult[i] ** 2 * np.sum(ratio[ks] / s_k[ks])

    # covariance from the shared parameter estimates (Mack 1993, total reserve formula)
    cov = 0.0
    for i in range(n):
        ks = np.arange(lag[i], n - 1)
        cov += ult[i] * ult[i + 1 :].sum() * np.sum(2.0 * ratio[ks] / s_k[ks])

    total_param = param.sum() + cov
    return MackResult(
        ldf=ldf,
        sigma2=sigma2,
        latest=cl.latest,
        ultimate=ult,
        reserve=ult - cl.latest,
        process_se=np.sqrt(proc),
        parameter_se=np.sqrt(param),
        se=np.sqrt(proc + param),
        total_se=float(np.sqrt(proc.sum() + total_param)),
        total_process_se=float(np.sqrt(proc.sum())),
        total_parameter_se=float(np.sqrt(total_param)),
    )
