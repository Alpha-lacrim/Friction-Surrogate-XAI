"""Run staged hyperparameter optimization for one configured dataset target."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.data import DataLoader
from friction_surrogate_xai.optimization import HyperparameterOptimizationRunner
from friction_surrogate_xai.preprocessing import PreprocessingPipelineFactory


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run staged hyperparameter optimization.")
    parser.add_argument("--dataset", required=True, help="Configured dataset key.")
    parser.add_argument("--target", required=True, help="Target column to optimize.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model keys. Defaults to all enabled models.",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    return parser.parse_args()


def main() -> None:
    """Run optimization for one dataset target."""
    args = parse_args()
    loaded = DataLoader().load(args.dataset)
    preprocessing_factory = PreprocessingPipelineFactory()
    X = preprocessing_factory.get_feature_frame(loaded)
    y = loaded.dataframe.loc[:, args.target]

    artifacts = HyperparameterOptimizationRunner(
        preprocessing_factory=preprocessing_factory,
    ).run(
        dataset_key=args.dataset,
        X=X,
        y=y,
        target_name=args.target,
        model_keys=tuple(args.models) if args.models else None,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"Optimization summary: {artifacts.summary_path}")


if __name__ == "__main__":
    main()
