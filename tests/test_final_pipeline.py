"""Tests for the final resumable orchestration pipeline."""

from __future__ import annotations

from pathlib import Path

from friction_surrogate_xai.pipelines import (
    FinalPipelineRunner,
    StageResult,
    load_final_pipeline_config,
)
from friction_surrogate_xai.pipelines.config import with_overrides


def _pipeline_config(tmp_path: Path, *, run_id: str = "test-run", force_rerun: bool = False):
    stages = {stage_name: False for stage_name in FinalPipelineRunner.STAGE_ORDER}
    stages.update(
        {
            "load_datasets": True,
            "preprocessing": True,
            "final_report": True,
        }
    )
    return with_overrides(
        load_final_pipeline_config(),
        run={
            "run_id": run_id,
            "root_dir": str(tmp_path),
            "resume": True,
            "force_rerun": force_rerun,
            "continue_on_error": True,
            "log_to_mlflow": True,
        },
        selection={
            "dataset_keys": ["dataset_0172"],
            "target_columns": ["wear rate"],
            "model_keys": ["ridge"],
        },
        stages=stages,
        mlflow={"enabled": True},
    )


def test_final_pipeline_continues_after_stage_failure_and_writes_report(tmp_path) -> None:
    runner = FinalPipelineRunner(config=_pipeline_config(tmp_path))

    runner.stage_handlers["load_datasets"] = lambda: StageResult(
        message="loaded",
        artifact_paths=["load_artifact.csv"],
        metadata={"component_mlflow": runner._component_mlflow()},
    )

    def failing_stage() -> StageResult:
        raise RuntimeError("synthetic stage failure")

    runner.stage_handlers["preprocessing"] = failing_stage

    artifacts = runner.run(log_to_mlflow=False)
    states = {row["name"]: row for row in artifacts.stage_rows}

    assert states["load_datasets"]["status"] == "completed"
    assert states["load_datasets"]["metadata"]["component_mlflow"] is False
    assert states["preprocessing"]["status"] == "failed"
    assert "synthetic stage failure" in states["preprocessing"]["message"]
    assert states["final_report"]["status"] == "completed"
    assert artifacts.state_path.exists()
    assert artifacts.final_report_path is not None
    assert artifacts.final_report_path.exists()
    assert "Failed Or Incomplete Stages" in artifacts.final_report_path.read_text(encoding="utf-8")


def test_final_pipeline_resume_skips_completed_stages(tmp_path) -> None:
    config = _pipeline_config(tmp_path, run_id="resume-test")
    calls = {"load_datasets": 0, "preprocessing": 0}

    runner = FinalPipelineRunner(config=config)
    runner.stage_handlers["load_datasets"] = lambda: _counted_result(calls, "load_datasets")
    runner.stage_handlers["preprocessing"] = lambda: _counted_result(calls, "preprocessing")
    first = runner.run(log_to_mlflow=False)

    resumed = FinalPipelineRunner(config=config)
    resumed.stage_handlers["load_datasets"] = lambda: _counted_result(calls, "load_datasets")
    resumed.stage_handlers["preprocessing"] = lambda: _counted_result(calls, "preprocessing")
    second = resumed.run(log_to_mlflow=False)

    assert calls == {"load_datasets": 1, "preprocessing": 1}
    assert first.state_path == second.state_path
    assert {row["name"]: row["status"] for row in second.stage_rows}["load_datasets"] == "completed"


def _counted_result(calls: dict[str, int], stage_name: str) -> StageResult:
    calls[stage_name] += 1
    return StageResult(
        message=f"{stage_name} ok",
        artifact_paths=[f"{stage_name}.txt"],
        metadata={"calls": calls[stage_name]},
    )
