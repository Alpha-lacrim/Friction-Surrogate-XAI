"""Final resumable orchestration pipeline."""

from __future__ import annotations

import importlib.util
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.data import DataLoader, LoadedDataset
from friction_surrogate_xai.discretization import DiscretizationWorkflow
from friction_surrogate_xai.discretization.config import (
    load_discretization_config,
    with_overrides as discretization_with_overrides,
)
from friction_surrogate_xai.eda import EDAReportGenerator, load_eda_config
from friction_surrogate_xai.eda.config import with_overrides as eda_with_overrides
from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename
from friction_surrogate_xai.evaluation import OverfittingAuditRunner
from friction_surrogate_xai.models import load_modeling_config
from friction_surrogate_xai.models.config import with_overrides as modeling_with_overrides
from friction_surrogate_xai.optimization import HyperparameterOptimizationRunner
from friction_surrogate_xai.optimization.config import (
    load_optimization_config,
    with_overrides as optimization_with_overrides,
)
from friction_surrogate_xai.pipelines.config import (
    FinalPipelineConfig,
    load_final_pipeline_config,
)
from friction_surrogate_xai.pipelines.mlflow_logging import FinalPipelineMLflowLogger
from friction_surrogate_xai.pipelines.reporting import FinalProjectReportWriter
from friction_surrogate_xai.pipelines.state import PipelineStateStore
from friction_surrogate_xai.preprocessing import (
    PreprocessingArtifactGenerator,
    load_preprocessing_config,
)
from friction_surrogate_xai.preprocessing.config import (
    with_overrides as preprocessing_with_overrides,
)
from friction_surrogate_xai.statistical_comparison import StatisticalComparisonRunner
from friction_surrogate_xai.statistical_comparison.config import (
    load_statistical_comparison_config,
    with_overrides as statistical_with_overrides,
)
from friction_surrogate_xai.uncertainty import UncertaintyReportGenerator
from friction_surrogate_xai.uncertainty.config import (
    load_uncertainty_config,
    with_overrides as uncertainty_with_overrides,
)
from friction_surrogate_xai.xai import XAIReportGenerator, load_xai_config
from friction_surrogate_xai.xai.config import with_overrides as xai_with_overrides


@dataclass(frozen=True)
class StageResult:
    """Result returned by one pipeline stage."""

    message: str
    artifact_paths: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FinalPipelineArtifacts:
    """Artifacts produced by the final orchestration pipeline."""

    run_id: str
    root_dir: Path
    state_path: Path
    final_report_path: Path | None
    stage_rows: list[dict[str, Any]]


class FinalPipelineRunner:
    """Coordinate the full research workflow with resumable stage state."""

    STAGE_ORDER = (
        "load_datasets",
        "preprocessing",
        "eda",
        "discretization",
        "training_evaluation",
        "optimization",
        "uncertainty",
        "xai",
        "statistical_comparison",
        "final_report",
    )

    def __init__(
        self,
        config: FinalPipelineConfig | None = None,
        data_loader: DataLoader | None = None,
    ) -> None:
        self.config = config or load_final_pipeline_config()
        self.data_loader = data_loader or DataLoader()
        self.root_dir = self._root_dir()
        self.state_store = PipelineStateStore(self.root_dir / "state" / "pipeline_state.json")
        self.loaded_datasets: dict[str, LoadedDataset] = {}
        self._log_to_mlflow_override: bool | None = None
        self.stage_handlers: dict[str, Callable[[], StageResult]] = {
            "load_datasets": self._stage_load_datasets,
            "preprocessing": self._stage_preprocessing,
            "eda": self._stage_eda,
            "discretization": self._stage_discretization,
            "training_evaluation": self._stage_training_evaluation,
            "optimization": self._stage_optimization,
            "uncertainty": self._stage_uncertainty,
            "xai": self._stage_xai,
            "statistical_comparison": self._stage_statistical_comparison,
            "final_report": self._stage_final_report,
        }
        self.mlflow_logger = FinalPipelineMLflowLogger(self.config.mlflow)

    def run(self, *, log_to_mlflow: bool | None = None) -> FinalPipelineArtifacts:
        """Run all configured stages."""
        self._log_to_mlflow_override = log_to_mlflow
        for stage_name in self.STAGE_ORDER:
            if not self.config.stages.get(stage_name, True):
                self.state_store.mark_skipped(stage_name, message="disabled_in_config")
                continue
            if self._should_skip_completed(stage_name):
                continue
            self._run_one_stage(stage_name)

        self._refresh_completed_final_report()
        stage_rows = self.state_store.stage_table()
        final_report_path = self._final_report_path_from_state(stage_rows)
        if self._mlflow_enabled():
            self.mlflow_logger.log_run(
                run_id=self.config.run_id,
                root_dir=self.root_dir,
                stage_rows=stage_rows,
            )
        return FinalPipelineArtifacts(
            run_id=self.config.run_id,
            root_dir=self.root_dir,
            state_path=self.state_store.path,
            final_report_path=final_report_path,
            stage_rows=stage_rows,
        )

    def _run_one_stage(self, stage_name: str) -> None:
        self.state_store.mark_running(stage_name)
        try:
            result = self.stage_handlers[stage_name]()
        except Exception as exc:  # pragma: no cover - exact component errors vary by environment.
            message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
            self.state_store.mark_failed(stage_name, message=message)
            if not self.config.continue_on_error:
                raise
            return
        self.state_store.mark_completed(
            stage_name,
            message=result.message,
            artifact_paths=result.artifact_paths,
            metadata=result.metadata,
        )

    def _stage_load_datasets(self) -> StageResult:
        loaded = self._load_selected_datasets()
        self.loaded_datasets = loaded
        return StageResult(
            message=f"Loaded {len(loaded)} dataset(s).",
            artifact_paths=[],
            metadata={key: list(value.dataframe.shape) for key, value in loaded.items()},
        )

    def _stage_preprocessing(self) -> StageResult:
        generator = PreprocessingArtifactGenerator(config=self._preprocessing_config())
        results = {
            dataset_key: generator.run_dataset(dataset_key, log_to_mlflow=self._component_mlflow())
            for dataset_key in self._dataset_keys()
        }
        artifact_paths = [
            str(result.artifacts.pipeline_path)
            for result in results.values()
        ]
        return StageResult(
            message=f"Generated preprocessing artifacts for {len(results)} dataset(s).",
            artifact_paths=artifact_paths,
            metadata={"dataset_count": len(results)},
        )

    def _stage_eda(self) -> StageResult:
        self._ensure_loaded()
        generator = EDAReportGenerator(config=self._eda_config(), data_loader=self.data_loader)
        artifacts = {
            dataset_key: generator.run_dataset(loaded, log_to_mlflow=self._component_mlflow())
            for dataset_key, loaded in self.loaded_datasets.items()
        }
        return StageResult(
            message=f"Generated EDA reports for {len(artifacts)} dataset(s).",
            artifact_paths=[str(artifact.root_dir) for artifact in artifacts.values()],
            metadata={"dataset_count": len(artifacts)},
        )

    def _stage_discretization(self) -> StageResult:
        workflow = DiscretizationWorkflow(
            config=self._discretization_config(),
            data_loader=self.data_loader,
        )
        options = dict(self.config.stage_options.get("discretization", {}))
        artifacts = workflow.run(
            dataset_keys=self._dataset_keys(),
            target_columns=self._target_columns(),
            explicit_model_keys=self._model_keys_for_component(),
            compare=options.get("compare", True),
            log_to_mlflow=self._component_mlflow(),
        )
        paths = [str(artifact.metadata_path) for artifact in artifacts.datasets.values()]
        paths.extend(str(comparison.root_dir) for comparison in artifacts.comparisons.values())
        return StageResult(
            message=(
                f"Generated {len(artifacts.datasets)} discrete dataset(s) and "
                f"{len(artifacts.comparisons)} comparison(s)."
            ),
            artifact_paths=paths,
            metadata={
                "dataset_count": len(artifacts.datasets),
                "comparison_count": len(artifacts.comparisons),
            },
        )

    def _stage_training_evaluation(self) -> StageResult:
        self._ensure_loaded()
        config = self._modeling_config()
        runner = OverfittingAuditRunner(config=config)
        options = dict(self.config.stage_options.get("training_evaluation", {}))
        artifacts = []
        for dataset_key, loaded in self.loaded_datasets.items():
            X = self._features(loaded)
            target_names = self._target_columns_for_dataset(loaded)
            if bool(options.get("run_single_output", True)):
                for target_name in target_names:
                    artifacts.append(
                        runner.run(
                            dataset_key=dataset_key,
                            X=X,
                            y=loaded.dataframe[target_name],
                            target_name=target_name,
                            model_keys=self._model_keys_for_component(),
                            strategy=str(options.get("strategy", "repeated_kfold")),
                            include_bootstrap=bool(options.get("include_bootstrap", False)),
                            include_nested=bool(options.get("include_nested", False)),
                            log_to_mlflow=self._component_mlflow(),
                        )
                    )
            if bool(options.get("run_multi_output", True)) and len(target_names) > 1:
                artifacts.append(
                    runner.run(
                        dataset_key=dataset_key,
                        X=X,
                        y=loaded.dataframe.loc[:, list(target_names)].copy(),
                        target_name=None,
                        model_keys=self._model_keys_for_component(),
                        strategy=str(options.get("strategy", "repeated_kfold")),
                        include_bootstrap=bool(options.get("include_bootstrap", False)),
                        include_nested=bool(options.get("include_nested", False)),
                        log_to_mlflow=self._component_mlflow(),
                    )
                )
        return StageResult(
            message=f"Ran training/evaluation audits for {len(artifacts)} dataset-target run(s).",
            artifact_paths=[str(artifact.root_dir) for artifact in artifacts],
            metadata={"run_count": len(artifacts)},
        )

    def _stage_optimization(self) -> StageResult:
        self._ensure_loaded()
        runner = HyperparameterOptimizationRunner(config=self._optimization_config())
        options = dict(self.config.stage_options.get("optimization", {}))
        artifacts = []
        for dataset_key, loaded in self.loaded_datasets.items():
            X = self._features(loaded)
            target_names = self._target_columns_for_dataset(loaded)
            if bool(options.get("run_single_output", True)):
                for target_name in target_names:
                    artifacts.append(
                        runner.run(
                            dataset_key=dataset_key,
                            X=X,
                            y=loaded.dataframe[target_name],
                            target_name=target_name,
                            model_keys=self._model_keys_for_component(),
                            log_to_mlflow=self._component_mlflow(),
                        )
                    )
            if bool(options.get("run_multi_output", True)) and len(target_names) > 1:
                artifacts.append(
                    runner.run(
                        dataset_key=dataset_key,
                        X=X,
                        y=loaded.dataframe.loc[:, list(target_names)].copy(),
                        target_name=None,
                        model_keys=self._model_keys_for_component(),
                        log_to_mlflow=self._component_mlflow(),
                    )
                )
        return StageResult(
            message=f"Ran optimization for {len(artifacts)} dataset-target run(s).",
            artifact_paths=[str(artifact.root_dir) for artifact in artifacts],
            metadata={"run_count": len(artifacts)},
        )

    def _stage_uncertainty(self) -> StageResult:
        self._ensure_loaded()
        runner = UncertaintyReportGenerator(config=self._uncertainty_config())
        artifacts = []
        model_keys = tuple(self.config.selection.get("uncertainty_models", ())) or None
        for dataset_key, loaded in self.loaded_datasets.items():
            X = self._features(loaded)
            for target_name in self._target_columns_for_dataset(loaded):
                artifacts.append(
                    runner.generate(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        X=X,
                        y=loaded.dataframe[target_name],
                        model_keys=model_keys,
                        log_to_mlflow=self._component_mlflow(),
                    )
                )
        return StageResult(
            message=f"Generated uncertainty reports for {len(artifacts)} dataset-target run(s).",
            artifact_paths=[str(artifact.root_dir) for artifact in artifacts],
            metadata={"run_count": len(artifacts)},
        )

    def _stage_xai(self) -> StageResult:
        self._ensure_loaded()
        runner = XAIReportGenerator(config=self._xai_config())
        artifacts = []
        for dataset_key, loaded in self.loaded_datasets.items():
            X = self._features(loaded)
            for target_name in self._target_columns_for_dataset(loaded):
                for model_key in self._xai_model_keys(dataset_key, target_name):
                    artifacts.append(
                        runner.generate(
                            dataset_key=dataset_key,
                            target_name=target_name,
                            model_key=model_key,
                            X=X,
                            y=loaded.dataframe[target_name],
                            log_to_mlflow=self._component_mlflow(),
                        )
                    )
        return StageResult(
            message=f"Generated XAI reports for {len(artifacts)} model-target run(s).",
            artifact_paths=[str(artifact.root_dir) for artifact in artifacts],
            metadata={"run_count": len(artifacts)},
        )

    def _stage_statistical_comparison(self) -> StageResult:
        config = self._statistical_config()
        artifacts = StatisticalComparisonRunner(config=config).run(
            log_to_mlflow=self._component_mlflow()
        )
        return StageResult(
            message="Generated statistical comparison reports.",
            artifact_paths=[str(artifacts.root_dir)],
            metadata={
                "score_rows": len(artifacts.score_table),
                "wilcoxon_tests": len(artifacts.wilcoxon),
                "friedman_tests": len(artifacts.friedman),
                "nemenyi_tests": len(artifacts.nemenyi),
            },
        )

    def _stage_final_report(self) -> StageResult:
        report = self._write_final_report()
        return StageResult(
            message="Generated final project report.",
            artifact_paths=[str(report.markdown_path), str(report.stage_table_path)],
            metadata={"report_path": str(report.markdown_path)},
        )

    def _ensure_loaded(self) -> None:
        if not self.loaded_datasets:
            self.loaded_datasets = self._load_selected_datasets()

    def _load_selected_datasets(self) -> dict[str, LoadedDataset]:
        return {
            dataset_key: self.data_loader.load(dataset_key)
            for dataset_key in self._dataset_keys()
        }

    def _dataset_keys(self) -> tuple[str, ...]:
        keys = self.config.selection.get("dataset_keys")
        if keys:
            return tuple(str(key) for key in keys)
        return self.data_loader.catalog.dataset_keys()

    def _target_columns(self) -> tuple[str, ...]:
        return tuple(str(target) for target in self.config.selection.get("target_columns", ()))

    def _target_columns_for_dataset(self, loaded: LoadedDataset) -> tuple[str, ...]:
        configured = self._target_columns()
        if configured:
            return tuple(target for target in configured if target in loaded.dataframe.columns)
        return loaded.report.metadata.target_columns

    def _features(self, loaded: LoadedDataset) -> pd.DataFrame:
        return loaded.dataframe.loc[:, list(loaded.report.metadata.feature_columns)].copy()

    def _model_keys_for_component(self) -> tuple[str, ...] | None:
        configured = self.config.selection.get("model_keys", "all")
        if configured == "all" or configured is None:
            return None
        return tuple(str(model_key) for model_key in configured)

    def _xai_model_keys(self, dataset_key: str, target_name: str) -> tuple[str, ...]:
        configured = self.config.stage_options.get("xai", {}).get("model_keys")
        if configured:
            return tuple(str(model_key) for model_key in configured)
        artifact_models = self._top_models_from_optimization(dataset_key, target_name)
        if artifact_models:
            return artifact_models
        fallback_models = self.config.selection.get("xai_fallback_models", ("ridge",))
        return tuple(str(model_key) for model_key in fallback_models)

    def _top_models_from_optimization(self, dataset_key: str, target_name: str) -> tuple[str, ...]:
        path = (
            project_root()
            / "reports"
            / "optimization"
            / dataset_key
            / sanitize_filename(target_name)
            / "tables"
            / "stage2_top_models.csv"
        )
        if not path.exists():
            return ()
        frame = pd.read_csv(path)
        return tuple(str(value) for value in frame.get("model_key", pd.Series(dtype=str)).head(3))

    def _modeling_config(self):
        overrides = dict(self.config.component_overrides.get("modeling", {}))
        base = load_modeling_config()
        return modeling_with_overrides(base, **overrides) if overrides else base

    def _preprocessing_config(self):
        overrides = dict(self.config.component_overrides.get("preprocessing", {}))
        base = load_preprocessing_config()
        return preprocessing_with_overrides(base, **overrides) if overrides else base

    def _eda_config(self):
        overrides = dict(self.config.component_overrides.get("eda", {}))
        base = load_eda_config()
        return eda_with_overrides(base, **overrides) if overrides else base

    def _discretization_config(self):
        overrides = dict(self.config.component_overrides.get("discretization", {}))
        base = load_discretization_config()
        return discretization_with_overrides(base, **overrides) if overrides else base

    def _optimization_config(self):
        overrides = dict(self.config.component_overrides.get("optimization", {}))
        base = load_optimization_config()
        return optimization_with_overrides(base, **overrides) if overrides else base

    def _uncertainty_config(self):
        overrides = dict(self.config.component_overrides.get("uncertainty", {}))
        base = load_uncertainty_config()
        return uncertainty_with_overrides(base, **overrides) if overrides else base

    def _xai_config(self):
        overrides = dict(self.config.component_overrides.get("xai", {}))
        base = load_xai_config()
        if self.config.stage_options.get("xai", {}).get("safe_optional_dependencies", True):
            optional_overrides = {}
            if importlib.util.find_spec("shap") is None:
                optional_overrides["shap"] = {"enabled": False}
            if importlib.util.find_spec("lime") is None:
                optional_overrides["lime"] = {"enabled": False}
            if optional_overrides:
                base = xai_with_overrides(base, **optional_overrides)
        return xai_with_overrides(base, **overrides) if overrides else base

    def _statistical_config(self):
        overrides = dict(self.config.component_overrides.get("statistical_comparison", {}))
        base = load_statistical_comparison_config()
        stage_options = dict(self.config.stage_options.get("statistical_comparison", {}))
        if "auto_discover" in stage_options:
            base = statistical_with_overrides(
                base,
                inputs={"auto_discover": bool(stage_options["auto_discover"])},
            )
        return statistical_with_overrides(base, **overrides) if overrides else base

    def _component_mlflow(self) -> bool:
        return self._mlflow_enabled()

    def _mlflow_enabled(self) -> bool:
        if self._log_to_mlflow_override is not None:
            return bool(self._log_to_mlflow_override)
        return bool(self.config.run.get("log_to_mlflow", True)) and bool(
            self.config.mlflow.get("enabled", True)
        )

    def _write_final_report(self):
        writer = FinalProjectReportWriter()
        return writer.write(
            run_id=self.config.run_id,
            state_rows=self.state_store.stage_table(),
            root_dir=self.root_dir,
            config_summary={
                "dataset_keys": ";".join(self._dataset_keys()),
                "target_columns": ";".join(self._target_columns()),
                "model_keys": self.config.selection.get("model_keys", "all"),
                "resume": self.config.resume,
                "continue_on_error": self.config.continue_on_error,
            },
        )

    def _refresh_completed_final_report(self) -> None:
        final_state = self.state_store.get("final_report")
        if final_state and final_state.status == "completed":
            self._write_final_report()

    def _should_skip_completed(self, stage_name: str) -> bool:
        if not self.config.resume or self.config.force_rerun:
            return False
        state = self.state_store.get(stage_name)
        return bool(state and state.status == "completed")

    def _root_dir(self) -> Path:
        configured = self.config.root_dir
        root = configured if configured.is_absolute() else project_root() / configured
        return ensure_directory(root / self.config.run_id)

    @staticmethod
    def _final_report_path_from_state(stage_rows: list[dict[str, Any]]) -> Path | None:
        for row in stage_rows:
            if row.get("name") == "final_report" and row.get("artifact_paths"):
                return Path(row["artifact_paths"][0])
        return None
