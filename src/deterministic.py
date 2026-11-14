"""Deterministic reserving methods implemented from scratch with numpy.

* Volume-weighted chain ladder (Mack 1993 estimator of the development factors)
* Exponential-decay tail factor (log-linear fit of ``ln(f - 1)`` on development age)
* Bornhuetter-Ferguson (1972) with an explicit a-priori expected loss ratio
* Cape Cod / Stanard-Buhlmann (expected loss ratio estimated from the triangle)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .triangle import is_observed, latest_diagonal, latest_lag


@dataclass(frozen=True)
class ChainLadderResult:
    """Output of :func:`chain_ladder`. Arrays are indexed by accident year."""

    ldf: np.ndarray  # age-to-age factors f_0..f_{n-2}
    tail: float
    cdf_by_lag: np.ndarray  # cumulative development factor to ultimate for each lag (incl. tail)
    latest: np.ndarray
    cdf: np.ndarray  # CDF applying to each accident year's latest diagonal
    ultimate: np.ndarray
    projected: np.ndarray  # completed square (no tail), observed cells kept

    @property
    def reserve(self) -> np.ndarray:
        """Outstanding (IBNR + IBNER or full unpaid, depending on the triangle) by accident year."""
        return self.ultimate - self.latest


def volume_weighted_ldf(cum: np.ndarray) -> np.ndarray:
    """Volume-weighted age-to-age factors ``f_j = sum_i C_{i,j+1} / sum_i C_ij``.

    Only accident years observed at both ages ``j`` and ``j+1`` contribute.
    """
    n_dev = cum.shape[1]
    ldf = np.empty(n_dev - 1)
    for j in range(n_dev - 1):
        both = is_observed(cum[:, j]) & is_observed(cum[:, j + 1])
        ldf[j] = cum[both, j + 1].sum() / cum[both, j].sum()
    return ldf


def cdf_from_ldf(ldf: np.ndarray, tail: float = 1.0) -> np.ndarray:
    """Cumulative factors to ultimate for every lag 0..n-1 (the last lag gets only the tail)."""
    with_tail = np.append(ldf, tail)
    return np.cumprod(with_tail[::-1])[::-1]


def exponential_tail(
    ldf: np.ndarray,
    start_lag: int = 2,
    extrap_periods: int = 100,
    threshold: float = 1.00001,
) -> tuple[float, float, float]:
    """Tail factor from an exponential decay fit of the development portion of the LDFs.

    Fits ``ln(f_k - 1) = a + b k`` by ordinary least squares on factors ``k >= start_lag``
    with ``f_k > threshold`` and extrapolates ``extrap_periods`` further factors.
    This mirrors ``chainladder.TailCurve(curve='exponential')``.

    Returns
    -------
    tail, intercept a, slope b
    """
    k = np.arange(len(ldf))
    use = (k >= start_lag) & (ldf > threshold)
    if use.sum() < 2:
        return 1.0, np.nan, np.nan
    b, a = np.polyfit(k[use], np.log(ldf[use] - 1.0), 1)
    future_k = np.arange(len(ldf), len(ldf) + extrap_periods)
    tail = float(np.prod(1.0 + np.exp(a + b * future_k)))
    return tail, float(a), float(b)


def project_square(cum: np.ndarray, ldf: np.ndarray) -> np.ndarray:
    """Fill the lower triangle by applying the development factors to the latest diagonal."""
    out = cum.astype(float).copy()
    for j in range(cum.shape[1] - 1):
        missing = ~is_observed(out[:, j + 1]) & is_observed(out[:, j])
        out[missing, j + 1] = out[missing, j] * ldf[j]
    return out


def chain_ladder(cum: np.ndarray, tail: float = 1.0, ldf: np.ndarray | None = None) -> ChainLadderResult:
    """Basic chain-ladder projection with volume-weighted factors and a constant tail."""
    ldf = volume_weighted_ldf(cum) if ldf is None else ldf
    cdf_lag = cdf_from_ldf(ldf, tail)
    latest = latest_diagonal(cum)
    cdf = cdf_lag[latest_lag(cum)]
    return ChainLadderResult(
        ldf=ldf,
        tail=tail,
        cdf_by_lag=cdf_lag,
        latest=latest,
        cdf=cdf,
        ultimate=latest * cdf,
        projected=project_square(cum, ldf),
    )


def bornhuetter_ferguson(latest: np.ndarray, cdf: np.ndarray, premium: np.ndarray, elr: np.ndarray | float) -> np.ndarray:
    """Bornhuetter-Ferguson ultimate: ``latest + premium * ELR * (1 - 1/CDF)``.

    The unreported (or unpaid) share ``1 - 1/CDF`` comes from the chain-ladder pattern,
    but it is applied to the a-priori expected losses instead of the actual emergence.
    """
    return latest + premium * elr * (1.0 - 1.0 / cdf)


def cape_cod_elr(latest: np.ndarray, cdf: np.ndarray, premium: np.ndarray) -> float:
    """Cape Cod (Stanard-Buhlmann) expected loss ratio: reported losses / "used-up" premium."""
    return float(latest.sum() / (premium / cdf).sum())


def cape_cod(latest: np.ndarray, cdf: np.ndarray, premium: np.ndarray) -> tuple[np.ndarray, float]:
    """Cape Cod ultimate (BF with the ELR estimated from all accident years, no trend, no decay)."""
    elr = cape_cod_elr(latest, cdf, premium)
    return bornhuetter_ferguson(latest, cdf, premium, elr), elr


def apriori_elr_from_mature_years(
    ultimate: np.ndarray, premium: np.ndarray, n_mature: int = 5
) -> float:
    """Premium-weighted average ultimate loss ratio of the ``n_mature`` oldest accident years.

    Used as the Bornhuetter-Ferguson a-priori ELR. Only mature years are used so the
    prior is not driven by the (highly leveraged) chain-ladder estimates of recent years.
    """
    return float(ultimate[:n_mature].sum() / premium[:n_mature].sum())
