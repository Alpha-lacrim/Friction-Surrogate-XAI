"""Command line entrypoint for the final orchestration pipeline."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.pipelines import FinalPipelineRunner, load_final_pipeline_config


def main() -> None:
    """Run the final orchestration pipeline."""
    parser = argparse.ArgumentParser(description="Run the resumable final ML pipeline.")
    parser.add_argument(
        "--config",
        default="configs/final_pipeline.yaml",
        help="Path to final pipeline YAML config.",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable pipeline MLflow logging.")
    args = parser.parse_args()

    artifacts = FinalPipelineRunner(config=load_final_pipeline_config(args.config)).run(
        log_to_mlflow=not args.no_mlflow
    )
    print(f"Pipeline state: {artifacts.state_path}")
    if artifacts.final_report_path:
        print(f"Final report: {artifacts.final_report_path}")


if __name__ == "__main__":
    main()
