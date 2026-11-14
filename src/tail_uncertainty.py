"""Tail-factor model risk: candidate tails, sensitivity and a tail-randomised bootstrap.

The tail beyond 120 months cannot be estimated from the triangle alone; different,
equally defensible curves give very different factors. Two views are produced:

* a deterministic sensitivity table (unpaid under each candidate tail);
* a simulation in which every bootstrap scenario draws its own tail. The tail
  development ``T - 1`` is lognormal with mean equal to the selected ``T - 1`` (so the
  best estimate does not move) and a log-sd calibrated so that the 5%-95% range spans
  the lowest and highest non-unity candidate:
  ``sigma = ln((T_hi - 1) / (T_lo - 1)) / (2 * 1.645)``.
  The tail draw is independent of the triangle scenario and common to all accident
  years (one tail assumption applies to the whole portfolio).

This is a judgement-based calibration, not an estimate of tail parameter error.
"""

from __future__ import annotations

import numpy as np

from .deterministic import exponential_tail, inverse_power_tail, volume_weighted_ldf

Z95 = 1.6448536269514722  # standard normal 95% quantile


def paid_tail_candidates(tri_paid: np.ndarray, tri_incurred: np.ndarray, start_lag: int = 2) -> dict[str, float]:
    """Paid tail factors under alternative, individually defensible assumptions."""
    f_p, f_i = volume_weighted_ldf(tri_paid), volume_weighted_ldf(tri_incurred)
    exp_p = exponential_tail(f_p, start_lag)[0]
    inv_p = inverse_power_tail(f_p, start_lag)[0]
    exp_i = max(exponential_tail(f_i, start_lag)[0], 1.0)
    inv_i = max(inverse_power_tail(f_i, start_lag)[0], 1.0)
    ratio = tri_incurred[0, -1] / tri_paid[0, -1]  # incurred / paid of the oldest year at the last age
    return {
        "none": 1.0,
        "paid_exponential": exp_p,
        "paid_inverse_power": inv_p,
        "inc_to_paid_x_inc_exponential": max(exp_p, ratio * exp_i),  # selected in the base case
        "inc_to_paid_x_inc_inverse_power": max(inv_p, ratio * inv_i),
    }


def lognormal_tail_sigma(candidates: dict[str, float]) -> float:
    """Log-sd of ``T - 1`` so that the 90% interval spans the non-unity candidates."""
    dev = np.array([t - 1.0 for t in candidates.values() if t > 1.0 + 1e-9])
    if len(dev) < 2:
        return 0.0
    return float(np.log(dev.max() / dev.min()) / (2.0 * Z95))


def simulate_tails(selected: float, sigma: float, n_sims: int, seed: int) -> np.ndarray:
    """Draw tail factors with ``E[T] = selected`` and lognormal ``T - 1``."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    return 1.0 + (selected - 1.0) * np.exp(sigma * z - 0.5 * sigma**2)


def unpaid_with_tail(latest: np.ndarray, reserves_by_origin: np.ndarray, tails: np.ndarray | float) -> np.ndarray:
    """Total unpaid per simulation when the projected ultimate is multiplied by the tail."""
    tails = np.broadcast_to(np.asarray(tails, dtype=float), (reserves_by_origin.shape[0],))
    ultimate = latest[None] + reserves_by_origin
    return (ultimate * tails[:, None] - latest[None]).sum(axis=1)
