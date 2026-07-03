"""Tests for discrete-input dataset generation and comparison."""

from __future__ import annotations

import pandas as pd

from friction_surrogate_xai.discretization.binning import DatasetDiscretizer
from friction_surrogate_xai.discretization.comparison import OriginalVsDiscreteComparator, TopModelSelector
from friction_surrogate_xai.discretization.config import load_discretization_config, with_overrides
from friction_surrogate_xai.discretization.workflow import DiscretizationWorkflow


def _frame(n_rows: int = 18) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "No.": list(range(1, n_rows + 1)),
            "Tool Shape": [(index % 3) + 1 for index in range(n_rows)],
            "Rotational Speed": [100 + 10 * index for index in range(n_rows)],
            "Plunging Speed": [1.0 + 0.25 * index for index in range(n_rows)],
            "Composite Volume Fraction (%)": [0 for _ in range(n_rows)],
            "wear rate": [0.4 * index + 0.1 * (index % 3) for index in range(n_rows)],
        }
    )


def _fast_config(tmp_path):
    return with_overrides(
        load_discretization_config(),
        output={
            "dataset_root_dir": str(tmp_path / "datasets"),
            "report_root_dir": str(tmp_path / "reports"),
            "save_csv": True,
            "save_excel": False,
        },
        comparison={
            "enabled": True,
            "target_columns": ["wear rate"],
            "cv_splits": 3,
            "repeated_seeds": [11],
            "top_n_models": 2,
            "optimization_root_dir": str(tmp_path / "optimization"),
            "fallback_top_models": ["ridge", "linear_regression"],
            "use_best_parameters_when_available": True,
        },
        mlflow={"enabled": False},
    )


def test_discretizer_converts_continuous_inputs_to_integer_bins() -> None:
    frame = _frame()
    discretizer = DatasetDiscretizer(
        continuous_features=(
            "Rotational Speed",
            "Plunging Speed",
            "Composite Volume Fraction (%)",
        ),
        method="quantile",
        n_bins=3,
    )

    result = discretizer.transform(frame, dataset_key="demo")

    for column in (
        "Rotational Speed",
        "Plunging Speed",
        "Composite Volume Fraction (%)",
    ):
        assert pd.api.types.is_integer_dtype(result.dataframe[column]), column
    assert result.dataframe["Composite Volume Fraction (%)"].nunique() == 1
    assert result.dataframe["Composite Volume Fraction (%)"].iloc[0] == 0
    assert result.dataframe["wear rate"].equals(frame["wear rate"])
    assert set(result.metadata["feature"]) == {
        "Rotational Speed",
        "Plunging Speed",
        "Composite Volume Fraction (%)",
    }


def test_top_model_selector_reads_optimization_artifacts(tmp_path) -> None:
    config = _fast_config(tmp_path)
    table_dir = tmp_path / "optimization" / "dataset_0172" / "wear_rate" / "tables"
    table_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"model_key": "random_forest", "params_json": '{"max_depth": 3}'},
            {"model_key": "ridge", "params_json": '{"alpha": 1.0}'},
            {"model_key": "elasticnet", "params_json": '{"alpha": 0.1}'},
        ]
    ).to_csv(table_dir / "stage2_top_models.csv", index=False)
    pd.DataFrame(
        [
            {"model_key": "random_forest", "params_json": '{"max_depth": 4}'},
            {"model_key": "ridge", "params_json": '{"alpha": 10.0}'},
        ]
    ).to_csv(table_dir / "best_parameters.csv", index=False)

    selected = TopModelSelector(config).select(
        dataset_key="dataset_0172",
        target_name="wear rate",
    )

    assert [model.model_key for model in selected] == ["random_forest", "ridge"]
    assert selected[0].params == {"max_depth": 4}
    assert selected[1].params == {"alpha": 10.0}
    assert {model.source for model in selected} == {"optimization_artifacts"}


def test_original_vs_discrete_comparison_uses_same_models_and_writes_reports(tmp_path) -> None:
    config = _fast_config(tmp_path)
    original = _frame()
    discrete = DatasetDiscretizer(
        continuous_features=("Rotational Speed", "Plunging Speed", "Composite Volume Fraction (%)"),
        method="quantile",
        n_bins=3,
    ).transform(original, "dataset_0172").dataframe

    artifacts = OriginalVsDiscreteComparator(config=config).compare(
        dataset_key="dataset_0172",
        original_dataframe=original,
        discrete_dataframe=discrete,
        feature_columns=(
            "Tool Shape",
            "Rotational Speed",
            "Plunging Speed",
            "Composite Volume Fraction (%)",
        ),
        target_name="wear rate",
        explicit_model_keys=("ridge", "linear_regression"),
        log_to_mlflow=False,
    )

    assert artifacts.score_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.markdown_path.exists()
    assert set(artifacts.variant_scores["variant"]) == {"original", "discrete"}
    assert set(artifacts.variant_scores["model_key"]) == {"ridge", "linear_regression"}
    assert {"objective_value_original", "objective_value_discrete"}.issubset(
        artifacts.comparison_summary.columns
    )


def test_workflow_can_generate_datasets_without_running_comparison(tmp_path) -> None:
    config = _fast_config(tmp_path)
    frame = _frame()

    class StubLoader:
        def load_all(self):
            from friction_surrogate_xai.data.contracts import LoadedDataset
            from types import SimpleNamespace

            metadata = SimpleNamespace(
                rows=len(frame),
                columns=len(frame.columns),
                feature_columns=(
                    "Tool Shape",
                    "Rotational Speed",
                    "Plunging Speed",
                    "Composite Volume Fraction (%)",
                ),
            )
            report = SimpleNamespace(metadata=metadata)
            return {
                "dataset_0172": LoadedDataset(
                    config=SimpleNamespace(key="dataset_0172"),
                    dataframe=frame,
                    report=report,
                )
            }

    artifacts = DiscretizationWorkflow(config=config, data_loader=StubLoader()).run(
        compare=False,
        log_to_mlflow=False,
    )

    generated = artifacts.datasets["dataset_0172"]
    assert generated.csv_path is not None
    assert generated.csv_path.exists()
    assert generated.excel_path is None
    assert generated.metadata_path.exists()
    assert artifacts.comparisons == {}
