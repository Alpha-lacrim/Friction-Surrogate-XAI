"""Static dataset metadata derived from the project specification and inspected files."""

from __future__ import annotations

from dataclasses import dataclass

ID_COLUMN = "No."
FEATURE_COLUMNS = (
    "Tool Shape",
    "Rotational Speed",
    "Plunging Speed",
    "Composite Volume Fraction (%)",
)
COMMON_TARGET_COLUMNS = (
    "Si Particle Size (um)",
    "Hardness (HV)",
    "wear rate",
    "Ultimate Compression Strength (MPa)",
    "Elongation (%)",
)
SPECIAL_TARGET_COLUMNS = {
    "dataset_0136": ("Temperature (°C)", "Strain"),
}


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata needed before implementing data loading or modeling."""

    key: str
    filename: str
    sheet_name: str
    rows: int
    columns: int
    constant_features: tuple[str, ...] = ()
    special_targets: tuple[str, ...] = ()


DATASET_SPECS = {
    "dataset_0136": DatasetSpec(
        key="dataset_0136",
        filename="Dataset 0136.xlsx",
        sheet_name="Extracted Data",
        rows=36,
        columns=12,
        constant_features=("Composite Volume Fraction (%)",),
        special_targets=("Temperature (°C)", "Strain"),
    ),
    "dataset_0172": DatasetSpec(
        key="dataset_0172",
        filename="Dataset 0172.xlsx",
        sheet_name="Extracted Data",
        rows=72,
        columns=10,
    ),
    "dataset_3772": DatasetSpec(
        key="dataset_3772",
        filename="Dataset 3772.xlsx",
        sheet_name="Extracted Data",
        rows=36,
        columns=10,
        constant_features=("Composite Volume Fraction (%)",),
    ),
}

