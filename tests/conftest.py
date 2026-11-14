"""Shared fixtures: triangles as numpy arrays and as chainladder.Triangle objects."""

from __future__ import annotations

import chainladder as cl
import numpy as np
import pytest

from src.data import load_segments, upper_triangle
from src.validation import premium_triangle, to_cl_triangle  # noqa: F401  (re-exported for tests)


def _cases() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]]:
    out = []
    for name in ("raa", "genins"):
        t = cl.load_sample(name)
        origins = t.origin.year.to_numpy()
        out.append((name, t.values[0, 0], origins, None))
    for key, seg in load_segments().items():
        out.append((f"{key}_paid", upper_triangle(seg.paid), seg.origins, seg.premium))
        out.append((f"{key}_incurred", upper_triangle(seg.incurred), seg.origins, seg.premium))
    return out


CASES = _cases()
CASE_IDS = [c[0] for c in CASES]


@pytest.fixture(params=CASES, ids=CASE_IDS)
def case(request):
    """(name, numpy triangle, origins, premium or None)."""
    return request.param


@pytest.fixture(params=[c for c in CASES if c[3] is not None], ids=[c[0] for c in CASES if c[3] is not None])
def premium_case(request):
    """Only the CAS segments, which carry earned premium."""
    return request.param
