"""Command line entrypoint for explainability reports."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.xai import XAIReportGenerator, load_xai_config


def main() -> None:
    """Run one configured explainability report."""
    parser = argparse.ArgumentParser(description="Generate SHAP/LIME XAI reports.")
    parser.add_argument("--dataset", default="dataset_0172", help="Configured dataset key.")
    parser.add_argument("--target", default="wear rate", help="Target column to explain.")
    parser.add_argument("--model", default="random_forest", help="Configured model key.")
    parser.add_argument("--config", default="configs/xai.yaml", help="Path to XAI YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    args = parser.parse_args()

    artifacts = XAIReportGenerator(config=load_xai_config(args.config)).generate(
        dataset_key=args.dataset,
        target_name=args.target,
        model_key=args.model,
        random_state=args.seed,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"Generated XAI report at: {artifacts.root_dir}")


if __name__ == "__main__":
    main()
