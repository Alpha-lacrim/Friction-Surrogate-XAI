"""Pytest configuration for source-layout imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

RAW_DATA_FILES = (
    ROOT / "data" / "raw" / "Dataset 0136.xlsx",
    ROOT / "data" / "raw" / "Dataset 0172.xlsx",
    ROOT / "data" / "raw" / "Dataset 3772.xlsx",
    ROOT / "data" / "raw" / "Mini project 1405.v2.pdf",
)


def raw_data_available() -> bool:
    """Return true when local-only raw assignment files are present."""
    return all(path.exists() and path.stat().st_size > 0 for path in RAW_DATA_FILES)


@pytest.fixture
def require_raw_data() -> None:
    """Skip tests that require local-only raw assignment files."""
    if not raw_data_available():
        pytest.skip("local raw assignment files are not present")
