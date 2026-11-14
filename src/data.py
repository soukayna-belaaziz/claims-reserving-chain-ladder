"""Loading the CAS Loss Reserve Database (NAIC Schedule P) triangles.

The raw file ``clrd2025.csv`` (accident years 1998-2007, ten development years,
full 10x10 squares, amounts in USD thousands, net of reinsurance) is shipped with
the ``chainladder`` Python package, so no network access is required.

Column conventions of the CAS database:

* ``CumPaidLoss``     cumulative paid losses and defence & cost containment (DCC)
* ``IncurredLosses``  incurred losses and DCC, *including* bulk and IBNR reserves
* ``BulkLoss``        bulk and IBNR reserves included in ``IncurredLosses``
* ``EarnedPremNet``   net earned premium of the accident year

The reported ("case incurred") triangle used throughout this project is
``IncurredLosses - BulkLoss``: paid + case reserves, i.e. the company's own IBNR
is removed so that the incurred chain ladder estimates IBNR independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEGMENTS_FILE = PROCESSED_DIR / "segments_long.csv"

#: Valuation year of the analysis: only cells with calendar year <= this are "known".
VALUATION_YEAR = 2007


@dataclass(frozen=True)
class SegmentSpec:
    """A company / line-of-business combination analysed in the project."""

    key: str
    grcode: int
    grname: str
    lob: str
    label: str


SEGMENTS: tuple[SegmentSpec, ...] = (
    SegmentSpec("wkcomp", 1767, "State Farm Mut Grp", "wkcomp", "Workers' compensation (State Farm Mut Grp)"),
    SegmentSpec("comauto", 2135, "Erie Ins Exchange Grp", "comauto", "Commercial auto liability (Erie Ins Exchange Grp)"),
    SegmentSpec("ppauto", 2003, "United Services Automobile Asn Grp", "ppauto", "Private passenger auto liability (USAA Grp)"),
)


@dataclass(frozen=True)
class Segment:
    """Full 10x10 squares for one segment (rows = accident year, cols = development lag)."""

    spec: SegmentSpec
    origins: np.ndarray  # accident years
    paid: np.ndarray  # cumulative paid, full square
    incurred: np.ndarray  # cumulative case incurred (paid + case reserves), full square
    premium: np.ndarray  # net earned premium by accident year

    @property
    def lags(self) -> np.ndarray:
        """Development ages in months (12, 24, ..., 120)."""
        return 12 * np.arange(1, self.paid.shape[1] + 1)


def raw_csv_path() -> Path:
    """Path of ``clrd2025.csv`` inside the installed chainladder package."""
    import chainladder

    return Path(chainladder.__file__).resolve().parent / "utils" / "data" / "clrd2025.csv"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CAS long-format file."""
    return pd.read_csv(path or raw_csv_path())


def extract_segments(raw: pd.DataFrame, specs: tuple[SegmentSpec, ...] = SEGMENTS) -> pd.DataFrame:
    """Return the long-format rows of the selected segments with a ``CaseIncurred`` column."""
    parts = []
    for s in specs:
        part = raw[(raw["GRCODE"] == s.grcode) & (raw["LOB"] == s.lob)].copy()
        if len(part) != 100:
            raise ValueError(f"{s.key}: expected 100 rows (10x10 square), found {len(part)}")
        part["segment"] = s.key
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["CaseIncurred"] = out["IncurredLosses"] - out["BulkLoss"]
    cols = ["segment", "GRCODE", "GRNAME", "LOB", "AccidentYear", "DevelopmentYear", "DevelopmentLag",
            "CumPaidLoss", "IncurredLosses", "BulkLoss", "CaseIncurred", "EarnedPremNet"]
    return out[cols].sort_values(["segment", "AccidentYear", "DevelopmentLag"]).reset_index(drop=True)


def _square(df: pd.DataFrame, value: str) -> np.ndarray:
    return df.pivot(index="AccidentYear", columns="DevelopmentLag", values=value).to_numpy(dtype=float)


def load_segments(path: Path = SEGMENTS_FILE) -> dict[str, Segment]:
    """Load the analysed segments, from the processed file if present, otherwise from the raw file."""
    long = pd.read_csv(path) if path.exists() else extract_segments(load_raw())
    out: dict[str, Segment] = {}
    for spec in SEGMENTS:
        df = long[long["segment"] == spec.key]
        prem = df.groupby("AccidentYear")["EarnedPremNet"].first().to_numpy(dtype=float)
        out[spec.key] = Segment(
            spec=spec,
            origins=np.sort(df["AccidentYear"].unique()),
            paid=_square(df, "CumPaidLoss"),
            incurred=_square(df, "CaseIncurred"),
            premium=prem,
        )
    return out


def upper_triangle(square: np.ndarray) -> np.ndarray:
    """Mask a full square to the cells known at the valuation date (row + col <= n - 1)."""
    n = square.shape[0]
    i, j = np.indices(square.shape)
    out = square.astype(float).copy()
    out[i + j > n - 1] = np.nan
    return out


def to_long_triangle(tri: np.ndarray, origins: np.ndarray) -> pd.DataFrame:
    """Convert a cumulative triangle to long format (origin, valuation year, value) for chainladder."""
    rows = []
    for i, o in enumerate(origins):
        for j in range(tri.shape[1]):
            if not np.isnan(tri[i, j]):
                rows.append({"origin": int(o), "valuation": int(o) + j, "value": tri[i, j]})
    return pd.DataFrame(rows)
