"""Command-line entrypoint for discrete dataset generation and comparison."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.discretization.workflow import DiscretizationWorkflow


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate discrete-input datasets and compare original vs discrete inputs."
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset keys.")
    parser.add_argument("--targets", nargs="*", default=None, help="Optional target columns.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit top-model keys. Defaults to optimization Top 3 or fallback.",
    )
    parser.add_argument("--skip-comparison", action="store_true", help="Only generate datasets.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    return parser.parse_args()


def main() -> None:
    """Run the configured discretization workflow."""
    args = parse_args()
    artifacts = DiscretizationWorkflow().run(
        dataset_keys=tuple(args.datasets) if args.datasets else None,
        target_columns=tuple(args.targets) if args.targets else None,
        explicit_model_keys=tuple(args.models) if args.models else None,
        compare=not args.skip_comparison,
        log_to_mlflow=not args.no_mlflow,
    )
    for dataset_key, dataset_artifacts in artifacts.datasets.items():
        print(f"{dataset_key}: {dataset_artifacts.csv_path or dataset_artifacts.excel_path}")
    for key, comparison in artifacts.comparisons.items():
        print(f"{key}: {comparison.markdown_path}")


if __name__ == "__main__":
    main()
