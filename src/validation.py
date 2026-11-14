"""Side-by-side comparison of the from-scratch implementation with the chainladder package."""

from __future__ import annotations

import chainladder as cl
import numpy as np
import pandas as pd

from .data import to_long_triangle
from .deterministic import bornhuetter_ferguson, cape_cod, chain_ladder, exponential_tail, volume_weighted_ldf
from .mack import mack


def to_cl_triangle(tri: np.ndarray, origins: np.ndarray) -> cl.Triangle:
    """Wrap a numpy cumulative triangle into a ``chainladder.Triangle`` (annual grain)."""
    long = to_long_triangle(tri, origins)
    long["origin"] = long["origin"].astype(str)
    long["valuation"] = long["valuation"].astype(str)
    return cl.Triangle(long, origin="origin", development="valuation", columns="value", cumulative=True)


def premium_triangle(premium: np.ndarray, origins: np.ndarray) -> cl.Triangle:
    """Premium vector as a single-column ``chainladder.Triangle`` (used as ``sample_weight``)."""
    df = pd.DataFrame({"origin": origins.astype(str), "valuation": str(int(origins.max())), "value": premium})
    return cl.Triangle(df, origin="origin", development="valuation", columns="value", cumulative=True)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    return float(np.max(np.abs(a[ok] - b[ok]) / np.abs(b[ok])))


def compare_with_package(tri: np.ndarray, origins: np.ndarray, premium: np.ndarray, elr: float) -> dict[str, float]:
    """Maximum relative differences (from-scratch vs chainladder 0.10.1) for one triangle."""
    t = to_cl_triangle(tri, origins)
    w = premium_triangle(premium, origins)
    res = chain_ladder(tri)
    dev = cl.Development().fit_transform(t)
    tail, _, _ = exponential_tail(res.ldf, start_lag=2)
    m = mack(tri)
    ref_mack = cl.MackChainladder().fit(cl.Development(sigma_interpolation="mack").fit_transform(t))
    cc_mine, _ = cape_cod(res.latest, res.cdf, premium)
    return {
        "ldf": _max_rel(volume_weighted_ldf(tri), dev.ldf_.values[0, 0, 0]),
        "cl_ultimate": _max_rel(res.ultimate, cl.Chainladder().fit(t).ultimate_.values[0, 0, :, -1]),
        "exp_tail": _max_rel(tail, cl.TailCurve(curve="exponential", fit_period=(36, None)).fit(dev).tail_.values),
        "mack_se_by_ay": _max_rel(m.se[1:], ref_mack.mack_std_err_.values[0, 0, 1:, -1]),
        "mack_total_se": _max_rel(m.total_se, ref_mack.total_mack_std_err_.values),
        "bf_ultimate": _max_rel(
            bornhuetter_ferguson(res.latest, res.cdf, premium, elr),
            cl.BornhuetterFerguson(apriori=elr).fit(t, sample_weight=w).ultimate_.values[0, 0, :, -1],
        ),
        "cape_cod_ultimate": _max_rel(
            cc_mine, cl.CapeCod(trend=0.0, decay=1.0).fit(t, sample_weight=w).ultimate_.values[0, 0, :, -1]
        ),
    }
