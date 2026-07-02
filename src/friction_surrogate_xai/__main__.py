"""Lightweight package health check."""

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.constants import PROJECT_NAME


def main() -> None:
    """Print basic package context without running experiments."""
    print(f"{PROJECT_NAME} skeleton is importable.")
    print(f"Project root: {project_root()}")


if __name__ == "__main__":
    main()

