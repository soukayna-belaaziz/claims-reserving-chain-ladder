"""Basic operations on run-off triangles stored as numpy arrays.

Convention: ``tri[i, j]`` is the cumulative amount of accident (origin) year ``i``
at development lag ``j`` (0-based). Unobserved cells are ``np.nan``. The calendar
(valuation) period of a cell is ``i + j``.
"""

from __future__ import annotations

import numpy as np


def is_observed(tri: np.ndarray) -> np.ndarray:
    """Boolean mask of observed cells."""
    return ~np.isnan(tri)


def to_incremental(cum: np.ndarray) -> np.ndarray:
    """Cumulative -> incremental (first column unchanged)."""
    inc = cum.astype(float).copy()
    inc[:, 1:] = cum[:, 1:] - cum[:, :-1]
    return inc


def to_cumulative(inc: np.ndarray) -> np.ndarray:
    """Incremental -> cumulative (NaN propagates to the right, as in a triangle)."""
    return np.cumsum(inc, axis=1)


def latest_diagonal(tri: np.ndarray) -> np.ndarray:
    """Last observed value of each origin row."""
    obs = is_observed(tri)
    last = obs.shape[1] - 1 - np.argmax(obs[:, ::-1], axis=1)
    return tri[np.arange(tri.shape[0]), last]


def latest_lag(tri: np.ndarray) -> np.ndarray:
    """0-based development lag of the latest observed cell of each row."""
    obs = is_observed(tri)
    return obs.shape[1] - 1 - np.argmax(obs[:, ::-1], axis=1)


def calendar_index(shape: tuple[int, int]) -> np.ndarray:
    """Calendar-period index ``i + j`` for every cell."""
    i, j = np.indices(shape)
    return i + j


def remove_last_diagonal(tri: np.ndarray) -> np.ndarray:
    """Drop the latest calendar diagonal (used for back-testing)."""
    out = tri.astype(float).copy()
    cal = calendar_index(tri.shape)
    last = np.nanmax(np.where(is_observed(tri), cal, -1))
    out[cal == last] = np.nan
    return out


def link_ratios(cum: np.ndarray) -> np.ndarray:
    """Individual age-to-age factors ``F_ij = C_{i,j+1} / C_ij``; shape (n, n-1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return cum[:, 1:] / cum[:, :-1]
