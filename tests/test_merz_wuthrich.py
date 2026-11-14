"""Merz-Wüthrich one-year CDR: published values and internal consistency with Mack."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.merz_wuthrich import merz_wuthrich

DATA = Path(__file__).parent / "data"


def _load(name: str) -> np.ndarray:
    return pd.read_csv(DATA / f"{name}.csv", index_col=0).to_numpy(dtype=float)


# R ChainLadder 0.2.x, vignette output of round(CDR(MackChainLadder(MW2014, est.sigma="Mack")), 1)
MW2014_CDR = [0.0, 0.4, 2.5, 16.7, 156.4, 137.7, 171.2, 70.3, 271.6, 310.1, 103.4, 632.6, 315.0, 406.1,
              285.2, 668.2, 733.2]
MW2014_MACK = [0.0, 0.4, 2.6, 16.9, 157.3, 207.2, 261.9, 292.3, 390.6, 502.1, 486.1, 806.9, 793.9, 891.7,
               916.5, 1106.1, 1295.7]


def test_mw2008_published_totals():
    """Merz & Wüthrich (2008), CAS E-Forum example: reserves 2,237,826, sqrt msep CDR 81,080, Mack 108,401."""
    res = merz_wuthrich(_load("mw2008"))
    assert res.reserve.sum() == pytest.approx(2_237_826, abs=1)
    assert res.cdr_total_se == pytest.approx(81_080, abs=1)
    assert res.mack_total_se == pytest.approx(108_401, abs=1)


def test_mw2014_matches_r_chainladder_by_accident_year():
    """17 x 17 triangle of Merz & Wüthrich (2014) against the R ChainLadder vignette (rounded to 0.1)."""
    res = merz_wuthrich(_load("mw2014"), linear_approximation=True)
    np.testing.assert_allclose(res.cdr_se, MW2014_CDR, atol=0.051)
    np.testing.assert_allclose(res.mack_se, MW2014_MACK, atol=0.051)
    assert res.reserve.sum() == pytest.approx(24_134.9, abs=0.05)
    assert res.cdr_total_se == pytest.approx(1_842.9, abs=0.05)
    assert res.mack_total_se == pytest.approx(3_233.7, abs=0.05)


def test_exact_and_linear_formulas_are_close():
    tri = _load("mw2014")
    exact, lin = merz_wuthrich(tri), merz_wuthrich(tri, linear_approximation=True)
    np.testing.assert_allclose(exact.cdr_se, lin.cdr_se, rtol=1e-3)
    assert exact.cdr_total_se >= lin.cdr_total_se  # prod(1+x)-1 >= sum(x) for x >= 0


def test_one_year_not_above_ultimate(case):
    """msep of the one-year CDR <= Mack msep of the ultimate, by accident year and in total."""
    _, tri, _, _ = case
    res = merz_wuthrich(tri)
    assert np.all(res.cdr_se <= res.mack_se * (1 + 1e-8))
    assert res.cdr_total_se <= res.mack_total_se
    assert 0.0 < res.one_year_share <= 1.0


def test_one_open_period_equals_mack(case):
    """With one development period left, all uncertainty emerges within the year: CDR s.e. = Mack s.e."""
    _, tri, _, _ = case
    res = merz_wuthrich(tri)
    assert res.cdr_se[0] == 0.0
    assert res.cdr_se[1] == pytest.approx(res.mack_se[1], rel=1e-8)


def test_rejects_non_square_triangle():
    with pytest.raises(ValueError):
        merz_wuthrich(_load("mw2008")[:-1])
