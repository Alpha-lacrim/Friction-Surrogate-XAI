"""Command-line health check for the evaluation framework."""

from __future__ import annotations

from friction_surrogate_xai.evaluation.config import load_evaluation_config


def main() -> None:
    """Print evaluation framework configuration without running model evaluation."""
    config = load_evaluation_config()
    print("Evaluation framework is importable.")
    print(f"Output root: {config.output_root}")
    print(f"Metrics: {', '.join(config.metrics.get('regression', []))}")
    print("Overfitting audit API: friction_surrogate_xai.evaluation.OverfittingAuditRunner")


if __name__ == "__main__":
    main()
