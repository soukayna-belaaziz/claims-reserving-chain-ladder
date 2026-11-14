"""On-levelled (trend-adjusted) a-priori loss ratios for Bornhuetter-Ferguson and Cape Cod.

The data contain no rate-change history, so the on-level adjustment is estimated from
the triangle itself, with information available at the valuation date only:

1. ``loss_ratio_trend``: weighted least squares of the log chain-ladder loss ratio on
   accident year. The weights are the "used-up" premium ``P_k / CDF_k`` (the Cape Cod
   credibility of each year), so immature, highly leveraged years count less. The slope
   is the net annual drift of the loss ratio (claims inflation minus rate changes and
   mix effects); it cannot separate the two.
2. ``trended_mature_prior``: the original BF prior (mature accident years) brought to the
   level of each accident year with that trend.
3. ``generalised_cape_cod``: Gluck (1997) Cape Cod with trend and decay; with
   ``decay = 1`` every year contributes, each loss ratio is trended to the target year.
   Identical to ``chainladder.CapeCod(trend, decay)`` (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deterministic import bornhuetter_ferguson


@dataclass(frozen=True)
class TrendFit:
    """Log-linear loss-ratio trend ``ln LR_k = a + b (k - k_last)``."""

    intercept: float
    slope: float
    slope_se: float
    n_years: int

    @property
    def annual_trend(self) -> float:
        """Annual multiplicative trend ``exp(b) - 1``."""
        return float(np.expm1(self.slope))

    @property
    def t_stat(self) -> float:
        """Slope divided by its standard error."""
        return self.slope / self.slope_se


def loss_ratio_trend(
    ultimate: np.ndarray,
    premium: np.ndarray,
    cdf: np.ndarray,
    origins: np.ndarray,
    exclude: tuple[int, ...] = (),
) -> TrendFit:
    """Weighted least-squares trend of log loss ratios; weights = used-up premium ``P / CDF``.

    ``exclude`` drops accident years with known exposure problems from the fit.
    """
    keep = ~np.isin(origins, exclude)
    x = (origins - origins.max()).astype(float)[keep]
    y = np.log(ultimate[keep] / premium[keep])
    w = (premium / cdf)[keep]
    design = np.column_stack([np.ones_like(x), x])
    xtw = design.T * w
    xtwx_inv = np.linalg.inv(xtw @ design)
    beta = xtwx_inv @ (xtw @ y)
    resid = y - design @ beta
    dof = len(y) - 2
    sigma2 = float((w * resid**2).sum() / dof)
    return TrendFit(float(beta[0]), float(beta[1]), float(np.sqrt(sigma2 * xtwx_inv[1, 1])), len(y))


def trended_mature_prior(
    ultimate: np.ndarray, premium: np.ndarray, trend: float, n_mature: int = 5
) -> np.ndarray:
    """A-priori ELR by accident year: mature-year loss ratios on-levelled to each target year.

    ``ELR_i = sum_{k mature} U_k (1 + trend)^(i - k) / sum_{k mature} P_k``.
    With ``trend = 0`` this is :func:`src.deterministic.apriori_elr_from_mature_years`.
    """
    idx = np.arange(len(ultimate))
    k = idx[:n_mature]
    factor = (1.0 + trend) ** (idx[:, None] - k[None])
    return (ultimate[k][None] * factor).sum(axis=1) / premium[k].sum()


def generalised_cape_cod(
    latest: np.ndarray,
    cdf: np.ndarray,
    premium: np.ndarray,
    trend: float = 0.0,
    decay: float = 1.0,
    weight_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cape Cod with trend and decay (Gluck 1997). Returns (ultimate, ELR by accident year).

    ``ELR_i = sum_k decay^|i-k| u_k LR_k (1+trend)^(i-k) / sum_k decay^|i-k| u_k`` with
    ``u_k = P_k / CDF_k`` and ``LR_k = latest_k / u_k``. ``weight_mask`` (bool) removes
    accident years from the ELR estimate (they still receive a BF ultimate).
    """
    n = len(latest)
    used_up = premium / cdf
    if weight_mask is not None:
        used_up = np.where(weight_mask, used_up, 0.0)
    lr = np.divide(latest, premium / cdf)
    i = np.arange(n)
    gap = i[:, None] - i[None]
    w = decay ** np.abs(gap) * used_up[None]
    elr = (w * lr[None] * (1.0 + trend) ** gap).sum(axis=1) / w.sum(axis=1)
    return bornhuetter_ferguson(latest, cdf, premium, elr), elr
