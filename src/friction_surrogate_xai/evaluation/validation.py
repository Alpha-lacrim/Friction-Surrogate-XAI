"""Cross-validation split strategies for tiny-data overfitting audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

from friction_surrogate_xai.models.config import ModelingConfig, load_modeling_config


@dataclass(frozen=True)
class FoldSplit:
    """One train/validation split."""

    strategy: str
    fold_id: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    seed: int | None = None
    repeat_id: int | None = None


@dataclass(frozen=True)
class NestedFoldSplit:
    """One nested CV outer split plus its inner splits."""

    outer_fold: FoldSplit
    inner_folds: tuple[FoldSplit, ...]


class ValidationStrategyFactory:
    """Create repeated KFold, LOOCV, nested CV, and bootstrap splits."""

    def __init__(self, config: ModelingConfig | None = None) -> None:
        self.config = config or load_modeling_config()

    def repeated_kfold(self, n_samples: int) -> tuple[FoldSplit, ...]:
        """Return repeated KFold splits across all configured seeds."""
        validation_config = self.config.validation
        repeated_config = dict(validation_config.get("repeated_kfold", {}))
        n_splits = self._bounded_splits(n_samples, int(validation_config.get("n_splits", 5)))
        n_repeats = int(repeated_config.get("n_repeats_per_seed", 1))
        shuffle = bool(validation_config.get("shuffle", True))

        folds: list[FoldSplit] = []
        fold_id = 0
        for seed in self.config.repeated_seeds:
            for repeat_id in range(n_repeats):
                splitter = KFold(
                    n_splits=n_splits,
                    shuffle=shuffle,
                    random_state=seed + repeat_id if shuffle else None,
                )
                for train_indices, validation_indices in splitter.split(np.arange(n_samples)):
                    folds.append(
                        FoldSplit(
                            strategy="repeated_kfold",
                            fold_id=fold_id,
                            train_indices=train_indices,
                            validation_indices=validation_indices,
                            seed=seed,
                            repeat_id=repeat_id,
                        )
                    )
                    fold_id += 1
        return tuple(folds)

    def loocv(self, n_samples: int) -> tuple[FoldSplit, ...]:
        """Return leave-one-out splits."""
        folds: list[FoldSplit] = []
        splitter = LeaveOneOut()
        for fold_id, (train_indices, validation_indices) in enumerate(
            splitter.split(np.arange(n_samples))
        ):
            folds.append(
                FoldSplit(
                    strategy="loocv",
                    fold_id=fold_id,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    seed=None,
                    repeat_id=None,
                )
            )
        return tuple(folds)

    def choose_primary(self, n_samples: int) -> tuple[FoldSplit, ...]:
        """Choose the configured primary strategy with LOOCV fallback for tiny data."""
        loocv_config = dict(self.config.validation.get("loocv", {}))
        if (
            bool(loocv_config.get("enabled", True))
            and n_samples <= int(loocv_config.get("max_samples_for_auto_fallback", 40))
        ):
            return self.loocv(n_samples)
        strategy = str(self.config.validation.get("primary_strategy", "repeated_kfold"))
        if strategy == "repeated_kfold":
            return self.repeated_kfold(n_samples)
        if strategy == "loocv":
            return self.loocv(n_samples)
        raise ValueError(f"Unsupported primary validation strategy: {strategy}")

    def nested_cv(self, n_samples: int) -> tuple[NestedFoldSplit, ...]:
        """Return nested CV split definitions with inner folds inside each outer train fold."""
        nested_config = dict(self.config.validation.get("nested_cv", {}))
        outer_splits = self._bounded_splits(n_samples, int(nested_config.get("outer_splits", 5)))
        inner_splits = int(nested_config.get("inner_splits", 3))
        shuffle = bool(nested_config.get("shuffle", True))
        seed = self.config.repeated_seeds[0]
        outer_splitter = KFold(n_splits=outer_splits, shuffle=shuffle, random_state=seed)

        nested: list[NestedFoldSplit] = []
        for outer_fold_id, (outer_train, outer_validation) in enumerate(
            outer_splitter.split(np.arange(n_samples))
        ):
            bounded_inner_splits = self._bounded_splits(len(outer_train), inner_splits)
            inner_splitter = KFold(
                n_splits=bounded_inner_splits,
                shuffle=shuffle,
                random_state=seed + outer_fold_id,
            )
            inner_folds: list[FoldSplit] = []
            for inner_fold_id, (inner_train_local, inner_validation_local) in enumerate(
                inner_splitter.split(outer_train)
            ):
                inner_folds.append(
                    FoldSplit(
                        strategy="nested_inner",
                        fold_id=inner_fold_id,
                        train_indices=outer_train[inner_train_local],
                        validation_indices=outer_train[inner_validation_local],
                        seed=seed,
                        repeat_id=outer_fold_id,
                    )
                )
            nested.append(
                NestedFoldSplit(
                    outer_fold=FoldSplit(
                        strategy="nested_outer",
                        fold_id=outer_fold_id,
                        train_indices=outer_train,
                        validation_indices=outer_validation,
                        seed=seed,
                        repeat_id=None,
                    ),
                    inner_folds=tuple(inner_folds),
                )
            )
        return tuple(nested)

    def bootstrap(self, n_samples: int) -> tuple[FoldSplit, ...]:
        """Return bootstrap train/OOB validation splits."""
        bootstrap_config = dict(self.config.validation.get("bootstrap", {}))
        n_iterations = int(bootstrap_config.get("n_iterations", 100))
        sample_fraction = float(bootstrap_config.get("sample_fraction", 1.0))
        require_oob = bool(bootstrap_config.get("require_oob_samples", True))
        train_size = max(1, int(round(n_samples * sample_fraction)))

        all_indices = np.arange(n_samples)
        folds: list[FoldSplit] = []
        for iteration, seed in self._bootstrap_seed_stream(n_iterations):
            rng = np.random.default_rng(seed)
            train_indices = rng.choice(all_indices, size=train_size, replace=True)
            validation_indices = np.setdiff1d(all_indices, np.unique(train_indices), assume_unique=False)
            if require_oob and len(validation_indices) == 0:
                continue
            folds.append(
                FoldSplit(
                    strategy="bootstrap_oob",
                    fold_id=iteration,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    seed=seed,
                    repeat_id=None,
                )
            )
        return tuple(folds)

    def split_summary(self, n_samples: int) -> dict[str, Any]:
        """Return a lightweight summary of configured split strategies."""
        nested = self.nested_cv(n_samples) if self.config.validation.get("nested_cv", {}).get("enabled", True) else ()
        return {
            "n_samples": n_samples,
            "primary_fold_count": len(self.choose_primary(n_samples)),
            "repeated_kfold_count": len(self.repeated_kfold(n_samples)),
            "loocv_count": len(self.loocv(n_samples)),
            "nested_outer_fold_count": len(nested),
            "nested_inner_fold_count": sum(len(split.inner_folds) for split in nested),
            "bootstrap_count": len(self.bootstrap(n_samples))
            if self.config.validation.get("bootstrap", {}).get("enabled", True)
            else 0,
        }

    def _bootstrap_seed_stream(self, n_iterations: int) -> Iterator[tuple[int, int]]:
        seeds = self.config.repeated_seeds
        for iteration in range(n_iterations):
            yield iteration, seeds[iteration % len(seeds)] + iteration

    @staticmethod
    def _bounded_splits(n_samples: int, requested_splits: int) -> int:
        if n_samples < 2:
            raise ValueError("At least two samples are required for validation splits.")
        return max(2, min(int(requested_splits), n_samples))
