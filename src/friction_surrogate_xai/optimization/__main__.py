"""Command-line health check for the optimization package."""

from __future__ import annotations

from friction_surrogate_xai.optimization.config import load_optimization_config


def main() -> None:
    """Print optimization configuration without running searches."""
    config = load_optimization_config()
    print("Optimization framework is importable.")
    print(f"Output root: {config.output_root}")
    print(f"Random Search trials/model: {config.stage1_random_search.get('n_trials_per_model')}")
    print(f"Optuna trials/model: {config.stage3_optuna.get('n_trials_per_model')}")
    print(f"Stage 2 top models: {config.stage2_selection.get('top_n_models')}")
    print("Grid Search is not part of this framework.")


if __name__ == "__main__":
    main()
