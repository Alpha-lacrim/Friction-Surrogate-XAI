"""Command line entrypoint for statistical model comparison."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.statistical_comparison import (
    StatisticalComparisonRunner,
    load_statistical_comparison_config,
)


def main() -> None:
    """Run statistical comparison reports."""
    parser = argparse.ArgumentParser(description="Generate statistical comparison reports.")
    parser.add_argument(
        "--config",
        default="configs/statistical_comparison.yaml",
        help="Path to statistical-comparison YAML config.",
    )
    parser.add_argument("--input", nargs="*", default=None, help="Optional score CSV paths.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    args = parser.parse_args()

    artifacts = StatisticalComparisonRunner(
        config=load_statistical_comparison_config(args.config),
    ).run(
        input_paths=tuple(args.input) if args.input else None,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"Generated statistical comparison report at: {artifacts.root_dir}")


if __name__ == "__main__":
    main()
