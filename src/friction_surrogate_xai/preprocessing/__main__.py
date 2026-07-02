"""Command-line entrypoint for preprocessing artifact generation."""

from __future__ import annotations

import argparse

from friction_surrogate_xai.preprocessing.runner import PreprocessingArtifactGenerator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate preprocessing pipeline artifacts.")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Generate local artifacts without logging them to MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate preprocessing artifacts for all configured datasets."""
    args = parse_args()
    results = PreprocessingArtifactGenerator().run_all(log_to_mlflow=not args.no_mlflow)
    for dataset_key, result in results.items():
        print(f"{dataset_key}: {result.artifacts.root_dir}")


if __name__ == "__main__":
    main()

