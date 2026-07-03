"""Nonparametric statistical tests for model comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon


@dataclass(frozen=True)
class StatisticalTestResult:
    """Tables produced by one statistical-comparison analysis."""

    wilcoxon: pd.DataFrame
    friedman: pd.DataFrame
    nemenyi: pd.DataFrame
    average_ranks: pd.DataFrame


class StatisticalComparator:
    """Run Wilcoxon, Friedman, and Nemenyi tests on matched score blocks."""

    def __init__(self, tests_config: dict[str, Any], alpha: float) -> None:
        self.config = tests_config
        self.alpha = alpha

    def compare(
        self,
        *,
        scores: pd.DataFrame,
        comparison_name: str,
        group_column: str,
        context_columns: tuple[str, ...],
        group_values: tuple[str, ...] | None = None,
    ) -> StatisticalTestResult:
        """Run all configured tests for one comparison family."""
        if scores.empty:
            return StatisticalTestResult(
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        filtered = scores.copy()
        if group_values:
            filtered = filtered.loc[filtered[group_column].isin(group_values)].copy()
        context_columns = tuple(column for column in context_columns if column in filtered.columns)
        if not context_columns:
            context_columns = ("comparison_type",)

        wilcoxon_rows: list[dict[str, Any]] = []
        friedman_rows: list[dict[str, Any]] = []
        nemenyi_frames: list[pd.DataFrame] = []
        rank_rows: list[dict[str, Any]] = []

        for context_values, group in filtered.groupby(list(context_columns), dropna=False):
            context = _context_dict(context_columns, context_values)
            pivot = self._pivot_scores(group, group_column)
            if pivot.shape[1] < 2:
                continue
            wilcoxon_rows.extend(
                self._wilcoxon_rows(
                    comparison_name=comparison_name,
                    context=context,
                    pivot=pivot,
                )
            )
            friedman_row = self._friedman_row(
                comparison_name=comparison_name,
                context=context,
                pivot=pivot,
            )
            if friedman_row:
                friedman_rows.append(friedman_row)
                nemenyi = self._nemenyi_table(
                    comparison_name=comparison_name,
                    context=context,
                    pivot=pivot,
                )
                if not nemenyi.empty:
                    nemenyi_frames.append(nemenyi)
            rank_rows.extend(
                self._average_rank_rows(
                    comparison_name=comparison_name,
                    context=context,
                    pivot=pivot,
                )
            )

        return StatisticalTestResult(
            wilcoxon=pd.DataFrame(wilcoxon_rows),
            friedman=pd.DataFrame(friedman_rows),
            nemenyi=pd.concat(nemenyi_frames, ignore_index=True)
            if nemenyi_frames
            else pd.DataFrame(),
            average_ranks=pd.DataFrame(rank_rows),
        )

    @staticmethod
    def _pivot_scores(group: pd.DataFrame, group_column: str) -> pd.DataFrame:
        pivot = group.pivot_table(
            index="block_id",
            columns=group_column,
            values="score",
            aggfunc="mean",
        )
        return pivot.dropna(axis=0, how="any").dropna(axis=1, how="all")

    def _wilcoxon_rows(
        self,
        *,
        comparison_name: str,
        context: dict[str, Any],
        pivot: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        config = dict(self.config.get("wilcoxon", {}))
        if not config.get("enabled", True):
            return []
        min_pairs = int(config.get("min_pairs", 3))
        rows: list[dict[str, Any]] = []
        for group_a, group_b in combinations(pivot.columns, 2):
            pair = pivot.loc[:, [group_a, group_b]].dropna()
            n_pairs = int(len(pair))
            if n_pairs < min_pairs:
                rows.append(
                    self._skipped_row(
                        comparison_name,
                        context,
                        group_a,
                        group_b,
                        "wilcoxon",
                        n_pairs,
                        f"fewer_than_{min_pairs}_pairs",
                    )
                )
                continue
            differences = pair[group_a] - pair[group_b]
            try:
                statistic, p_value = wilcoxon(
                    pair[group_a],
                    pair[group_b],
                    zero_method=str(config.get("zero_method", "wilcox")),
                    correction=bool(config.get("correction", False)),
                    alternative=str(config.get("alternative", "two-sided")),
                )
                status = "ok"
            except ValueError as exc:
                statistic, p_value, status = np.nan, np.nan, str(exc)
            rows.append(
                {
                    "comparison_name": comparison_name,
                    **context,
                    "test": "wilcoxon_signed_rank",
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_pairs": n_pairs,
                    "statistic": statistic,
                    "p_value": p_value,
                    "alpha": self.alpha,
                    "significant": bool(np.isfinite(p_value) and p_value < self.alpha),
                    "mean_group_a": float(pair[group_a].mean()),
                    "mean_group_b": float(pair[group_b].mean()),
                    "mean_difference_a_minus_b": float(differences.mean()),
                    "effect_direction": _direction(differences.mean()),
                    "status": status,
                }
            )
        return rows

    def _friedman_row(
        self,
        *,
        comparison_name: str,
        context: dict[str, Any],
        pivot: pd.DataFrame,
    ) -> dict[str, Any] | None:
        config = dict(self.config.get("friedman", {}))
        if not config.get("enabled", True):
            return None
        min_blocks = int(config.get("min_blocks", 3))
        min_groups = int(config.get("min_groups", 3))
        complete = pivot.dropna()
        n_blocks, n_groups = complete.shape
        row = {
            "comparison_name": comparison_name,
            **context,
            "test": "friedman",
            "n_blocks": int(n_blocks),
            "n_groups": int(n_groups),
            "groups": ";".join(map(str, complete.columns)),
            "alpha": self.alpha,
        }
        if n_blocks < min_blocks or n_groups < min_groups:
            row.update(
                {
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "status": f"requires_at_least_{min_blocks}_blocks_and_{min_groups}_groups",
                }
            )
            return row
        try:
            statistic, p_value = friedmanchisquare(
                *[complete[column].to_numpy(dtype=float) for column in complete.columns]
            )
            status = "ok"
        except ValueError as exc:
            statistic, p_value, status = np.nan, np.nan, str(exc)
        row.update(
            {
                "statistic": statistic,
                "p_value": p_value,
                "significant": bool(np.isfinite(p_value) and p_value < self.alpha),
                "status": status,
            }
        )
        return row

    def _nemenyi_table(
        self,
        *,
        comparison_name: str,
        context: dict[str, Any],
        pivot: pd.DataFrame,
    ) -> pd.DataFrame:
        config = dict(self.config.get("nemenyi", {}))
        if not config.get("enabled", True):
            return pd.DataFrame()
        complete = pivot.dropna()
        min_blocks = int(config.get("min_blocks", 3))
        min_groups = int(config.get("min_groups", 3))
        if complete.shape[0] < min_blocks or complete.shape[1] < min_groups:
            return pd.DataFrame()

        p_values = _nemenyi_p_values(complete)
        rows: list[dict[str, Any]] = []
        for group_a, group_b in combinations(p_values.index, 2):
            p_value = float(p_values.loc[group_a, group_b])
            rows.append(
                {
                    "comparison_name": comparison_name,
                    **context,
                    "test": "nemenyi_post_hoc",
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_blocks": int(complete.shape[0]),
                    "n_groups": int(complete.shape[1]),
                    "p_value": p_value,
                    "alpha": self.alpha,
                    "significant": bool(np.isfinite(p_value) and p_value < self.alpha),
                    "status": "ok",
                }
            )
        return pd.DataFrame(rows)

    def _average_rank_rows(
        self,
        *,
        comparison_name: str,
        context: dict[str, Any],
        pivot: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        complete = pivot.dropna()
        if complete.empty:
            return []
        ranks = complete.apply(lambda row: rankdata(-row.to_numpy(dtype=float)), axis=1)
        rank_frame = pd.DataFrame(
            list(ranks),
            columns=complete.columns,
            index=complete.index,
        )
        rows: list[dict[str, Any]] = []
        for group_name in rank_frame.columns:
            rows.append(
                {
                    "comparison_name": comparison_name,
                    **context,
                    "group": group_name,
                    "n_blocks": int(len(rank_frame)),
                    "mean_score": float(complete[group_name].mean()),
                    "std_score": float(complete[group_name].std(ddof=1))
                    if len(complete) > 1
                    else 0.0,
                    "average_rank": float(rank_frame[group_name].mean()),
                    "rank_direction": "lower_rank_is_better",
                }
            )
        return rows

    def _skipped_row(
        self,
        comparison_name: str,
        context: dict[str, Any],
        group_a: Any,
        group_b: Any,
        test: str,
        n_pairs: int,
        status: str,
    ) -> dict[str, Any]:
        return {
            "comparison_name": comparison_name,
            **context,
            "test": test,
            "group_a": group_a,
            "group_b": group_b,
            "n_pairs": n_pairs,
            "statistic": np.nan,
            "p_value": np.nan,
            "alpha": self.alpha,
            "significant": False,
            "mean_group_a": np.nan,
            "mean_group_b": np.nan,
            "mean_difference_a_minus_b": np.nan,
            "effect_direction": "unknown",
            "status": status,
        }


def _nemenyi_p_values(complete_scores: pd.DataFrame) -> pd.DataFrame:
    try:
        import scikit_posthocs as sp

        return sp.posthoc_nemenyi_friedman(complete_scores)
    except ImportError:
        return _nemenyi_p_values_fallback(complete_scores)


def _nemenyi_p_values_fallback(complete_scores: pd.DataFrame) -> pd.DataFrame:
    n_blocks, n_groups = complete_scores.shape
    rank_rows = complete_scores.apply(lambda row: rankdata(-row.to_numpy(dtype=float)), axis=1)
    rank_frame = pd.DataFrame(list(rank_rows), columns=complete_scores.columns)
    average_ranks = rank_frame.mean(axis=0)
    standard_error = np.sqrt(n_groups * (n_groups + 1) / (6.0 * n_blocks))
    p_values = pd.DataFrame(
        np.ones((n_groups, n_groups)),
        index=complete_scores.columns,
        columns=complete_scores.columns,
    )
    for group_a, group_b in combinations(complete_scores.columns, 2):
        q_statistic = abs(average_ranks[group_a] - average_ranks[group_b]) / standard_error
        p_value = float(studentized_range.sf(q_statistic * np.sqrt(2.0), n_groups, np.inf))
        p_values.loc[group_a, group_b] = p_value
        p_values.loc[group_b, group_a] = p_value
    return p_values


def _context_dict(columns: tuple[str, ...], values: Any) -> dict[str, Any]:
    if not isinstance(values, tuple):
        values = (values,)
    return dict(zip(columns, values, strict=False))


def _direction(mean_difference: float) -> str:
    if not np.isfinite(mean_difference) or np.isclose(mean_difference, 0.0):
        return "no_clear_direction"
    return "group_a_higher" if mean_difference > 0 else "group_b_higher"
