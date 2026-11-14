"""Prepare the data used by the project.

The CAS Loss Reserve Database file ``clrd2025.csv`` (5.9 MB, too large to commit) is
bundled with the ``chainladder`` package (pinned in requirements.txt), so this script
works offline. If the package file is missing, it is downloaded from the
chainladder GitHub repository into ``data/raw/``.

Output: ``data/processed/segments_long.csv`` (300 rows: 3 segments x 10 x 10 cells).

Run from the project root:  python data/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import SEGMENTS_FILE, extract_segments, load_raw, raw_csv_path  # noqa: E402

MIRROR_URL = (
    "https://raw.githubusercontent.com/casact/chainladder-python/master/"
    "chainladder/utils/data/clrd2025.csv"
)


def locate_raw() -> Path:
    """Return a local path to clrd2025.csv, downloading it only if the package copy is absent."""
    try:
        path = raw_csv_path()
        if path.exists():
            return path
    except ImportError:
        pass
    target = ROOT / "data" / "raw" / "clrd2025.csv"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {MIRROR_URL}")
        urllib.request.urlretrieve(MIRROR_URL, target)
    return target


def main() -> None:
    raw_path = locate_raw()
    print(f"Raw file: {raw_path}")
    raw = load_raw(raw_path)
    long = extract_segments(raw)
    SEGMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(SEGMENTS_FILE, index=False)
    print(f"Wrote {SEGMENTS_FILE.relative_to(ROOT)} ({len(long)} rows)")


if __name__ == "__main__":
    main()
