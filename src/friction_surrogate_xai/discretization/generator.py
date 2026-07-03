"""Generate discrete-input datasets and metadata artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.data import DataLoader, LoadedDataset
from friction_surrogate_xai.discretization.binning import DatasetDiscretizer
from friction_surrogate_xai.discretization.config import (
    DiscretizationConfig,
    load_discretization_config,
)
from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename, write_csv
from friction_surrogate_xai.evaluation.reports import EvaluationReportWriter


@dataclass(frozen=True)
class DiscreteDatasetArtifacts:
    """Generated artifacts for one discrete dataset."""

    dataset_key: str
    dataframe: pd.DataFrame
    csv_path: Path | None
    excel_path: Path | None
    metadata_path: Path
    comparison_path: Path
    summary_path: Path
    metadata: pd.DataFrame
    feature_comparison: pd.DataFrame


class DiscreteDatasetGenerator:
    """Create configured discrete-input datasets from loaded originals."""

    def __init__(
        self,
        config: DiscretizationConfig | None = None,
        data_loader: DataLoader | None = None,
    ) -> None:
        self.config = config or load_discretization_config()
        self.data_loader = data_loader or DataLoader()
        self.discretizer = DatasetDiscretizer(
            continuous_features=tuple(self.config.columns.get("continuous_input_features", ())),
            method=str(self.config.binning.get("method", "quantile")),
            n_bins=int(self.config.binning.get("n_bins", 3)),
            labels_start_at=int(self.config.binning.get("labels_start_at", 0)),
            include_lowest=bool(self.config.binning.get("include_lowest", True)),
            duplicates=str(self.config.binning.get("duplicates", "drop")),
            preserve_low_cardinality_integer_values=bool(
                self.config.binning.get("preserve_low_cardinality_integer_values", True)
            ),
            low_cardinality_unique_threshold=int(
                self.config.binning.get("low_cardinality_unique_threshold", 3)
            ),
            constant_bin_value=int(self.config.binning.get("constant_bin_value", 0)),
        )
        self.markdown = EvaluationReportWriter(self.config.reports)

    def run_all(self) -> dict[str, DiscreteDatasetArtifacts]:
        """Generate discrete-input datasets for all configured originals."""
        return {
            dataset_key: self.run_dataset(loaded_dataset)
            for dataset_key, loaded_dataset in self.data_loader.load_all().items()
        }

    def run_dataset(self, loaded_dataset: LoadedDataset) -> DiscreteDatasetArtifacts:
        """Generate one discrete-input dataset from a loaded original dataset."""
        dataset_key = loaded_dataset.key
        result = self.discretizer.transform(loaded_dataset.dataframe, dataset_key=dataset_key)

        dataset_dir = ensure_directory(self._dataset_root())
        metadata_dir = ensure_directory(self._report_root() / self.config.output.get("metadata_dir_name", "metadata"))
        markdown_dir = ensure_directory(
            self._report_root() / self.config.output.get("markdown_dir_name", "markdown")
        )

        csv_path = None
        excel_path = None
        base_name = f"{sanitize_filename(dataset_key)}_discrete"
        if self.config.output.get("save_csv", True):
            csv_path = dataset_dir / f"{base_name}.csv"
            result.dataframe.to_csv(csv_path, index=False, encoding="utf-8")
        if self.config.output.get("save_excel", True):
            excel_path = dataset_dir / f"{base_name}.xlsx"
            result.dataframe.to_excel(excel_path, index=False)

        metadata_path = write_csv(result.metadata, metadata_dir / f"{base_name}_metadata.csv")
        comparison_path = write_csv(
            result.comparison,
            metadata_dir / f"{base_name}_feature_comparison.csv",
        )
        summary_path = self._write_summary(
            dataset_key=dataset_key,
            loaded_dataset=loaded_dataset,
            metadata=result.metadata,
            comparison=result.comparison,
            output_path=markdown_dir / f"{base_name}_summary.md",
        )

        return DiscreteDatasetArtifacts(
            dataset_key=dataset_key,
            dataframe=result.dataframe,
            csv_path=csv_path,
            excel_path=excel_path,
            metadata_path=metadata_path,
            comparison_path=comparison_path,
            summary_path=summary_path,
            metadata=result.metadata,
            feature_comparison=result.comparison,
        )

    def _write_summary(
        self,
        *,
        dataset_key: str,
        loaded_dataset: LoadedDataset,
        metadata: pd.DataFrame,
        comparison: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        lines = [
            f"# Discrete Dataset Summary: {dataset_key}",
            "",
            "## Dataset",
            "",
            f"- Source rows: {loaded_dataset.report.metadata.rows}",
            f"- Source columns: {loaded_dataset.report.metadata.columns}",
            f"- Discretization method: `{self.config.binning.get('method', 'quantile')}`",
            f"- Requested bins: {self.config.binning.get('n_bins', 3)}",
            "",
            "## Discretized Features",
            "",
            self.markdown._markdown_table(metadata),
            "",
            "## Original vs Discrete Feature Summary",
            "",
            self.markdown._markdown_table(comparison),
            "",
        ]
        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _dataset_root(self) -> Path:
        configured = Path(self.config.output["dataset_root_dir"])
        return configured if configured.is_absolute() else project_root() / configured

    def _report_root(self) -> Path:
        configured = Path(self.config.output["report_root_dir"])
        return configured if configured.is_absolute() else project_root() / configured
