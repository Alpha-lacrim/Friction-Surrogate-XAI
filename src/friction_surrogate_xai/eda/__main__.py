"""Command-line entrypoint for automated EDA generation."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.eda.runner import EDAReportGenerator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate research-grade EDA reports.")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Generate local artifacts without logging them to MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    """Run EDA for all configured datasets."""
    args = parse_args()
    artifacts = EDAReportGenerator().run_all(log_to_mlflow=not args.no_mlflow)
    for dataset_key, dataset_artifacts in artifacts.items():
        print(f"{dataset_key}: {dataset_artifacts.root_dir}")


if __name__ == "__main__":
    main()

