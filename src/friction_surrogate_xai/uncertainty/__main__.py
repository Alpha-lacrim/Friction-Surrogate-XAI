"""Command line entrypoint for uncertainty estimation."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.uncertainty import UncertaintyReportGenerator, load_uncertainty_config


def main() -> None:
    """Run one uncertainty report."""
    parser = argparse.ArgumentParser(description="Generate uncertainty estimation reports.")
    parser.add_argument("--dataset", default="dataset_0172", help="Configured dataset key.")
    parser.add_argument("--target", default="wear rate", help="Target column.")
    parser.add_argument("--models", nargs="*", default=None, help="Configured model keys.")
    parser.add_argument("--config", default="configs/uncertainty.yaml", help="Path to YAML config.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    args = parser.parse_args()

    artifacts = UncertaintyReportGenerator(
        config=load_uncertainty_config(args.config),
    ).generate(
        dataset_key=args.dataset,
        target_name=args.target,
        model_keys=tuple(args.models) if args.models else None,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"Generated uncertainty report at: {artifacts.root_dir}")


if __name__ == "__main__":
    main()
