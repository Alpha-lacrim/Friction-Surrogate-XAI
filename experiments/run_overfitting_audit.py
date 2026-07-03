"""Run a leakage-safe overfitting audit for configured datasets."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.data import DataLoader
from friction_surrogate_xai.evaluation import OverfittingAuditRunner
from friction_surrogate_xai.preprocessing import PreprocessingPipelineFactory


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run overfitting audit reports.")
    parser.add_argument("--dataset", required=True, help="Configured dataset key.")
    parser.add_argument("--target", required=True, help="Target column to audit.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model keys. Defaults to all enabled models.",
    )
    parser.add_argument(
        "--strategy",
        default="repeated_kfold",
        choices=["primary", "repeated_kfold", "loocv", "nested_cv", "bootstrap_oob"],
        help="Primary validation strategy.",
    )
    parser.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap OOB audit.")
    parser.add_argument("--no-nested", action="store_true", help="Disable nested outer-fold audit.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    return parser.parse_args()


def main() -> None:
    """Run an overfitting audit for one dataset target."""
    args = parse_args()
    loaded = DataLoader().load(args.dataset)
    preprocessing_factory = PreprocessingPipelineFactory()
    X = preprocessing_factory.get_feature_frame(loaded)
    y = loaded.dataframe.loc[:, args.target]

    artifacts = OverfittingAuditRunner(preprocessing_factory=preprocessing_factory).run(
        dataset_key=args.dataset,
        X=X,
        y=y,
        target_name=args.target,
        model_keys=tuple(args.models) if args.models else None,
        strategy=args.strategy,
        include_bootstrap=not args.no_bootstrap,
        include_nested=not args.no_nested,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"Overfitting report: {artifacts.report_path}")


if __name__ == "__main__":
    main()
