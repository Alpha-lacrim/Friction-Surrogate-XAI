"""Tests for statistical model comparison."""

from __future__ import annotations

import pandas as pd

from friction_surrogate_xai.statistical_comparison import StatisticalComparisonRunner
from friction_surrogate_xai.statistical_comparison.config import (
    load_statistical_comparison_config,
    with_overrides,
)
from friction_surrogate_xai.statistical_comparison.data import normalize_score_table


def _score_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_offsets = {
        "random_forest": 0.12,
        "ridge": 0.06,
        "elasticnet": 0.00,
    }
    variant_offsets = {
        "original": 0.02,
        "discrete": -0.02,
    }
    output_offsets = {
        "single_output": 0.03,
        "multi_output": -0.01,
    }
    for block in range(8):
        base = 0.55 + 0.01 * block
        for model_key, model_offset in model_offsets.items():
            for variant, variant_offset in variant_offsets.items():
                for output_mode, output_offset in output_offsets.items():
                    rows.append(
                        {
                            "comparison_source": "synthetic",
                            "comparison_type": "synthetic_cv",
                            "dataset_key": "dataset_0172",
                            "target_name": "wear rate",
                            "model_key": model_key,
                            "variant": variant,
                            "output_mode": output_mode,
                            "block_id": str(block),
                            "metric": "r2",
                            "score": base + model_offset + variant_offset + output_offset,
                        }
                    )
    return pd.DataFrame(rows)


def _fast_config(tmp_path):
    return with_overrides(
        load_statistical_comparison_config(),
        output={"root_dir": str(tmp_path)},
        inputs={"auto_discover": False, "explicit_score_paths": []},
        mlflow={"enabled": False},
    )


def test_statistical_comparison_runner_generates_all_required_reports(tmp_path) -> None:
    artifacts = StatisticalComparisonRunner(config=_fast_config(tmp_path)).run(
        score_table=_score_table(),
        log_to_mlflow=False,
    )

    table_names = {path.name for path in artifacts.table_paths}
    figure_names = {path.name for path in artifacts.figure_paths}

    assert "normalized_scores.csv" in table_names
    assert "wilcoxon_signed_rank.csv" in table_names
    assert "friedman_test.csv" in table_names
    assert "nemenyi_post_hoc.csv" in table_names
    assert "average_ranks.csv" in table_names
    assert "significant_findings.csv" in table_names
    assert artifacts.markdown_paths[0].exists()
    assert any(name.endswith(".png") for name in figure_names)

    assert {"top_models", "original_vs_discrete", "single_vs_multi_output"}.issubset(
        set(artifacts.wilcoxon["comparison_name"])
    )
    assert (artifacts.wilcoxon["test"] == "wilcoxon_signed_rank").all()
    assert not artifacts.friedman.empty
    assert not artifacts.nemenyi.empty
    assert artifacts.wilcoxon["significant"].any()
    assert artifacts.nemenyi["significant"].any()


def test_normalize_score_table_accepts_existing_metric_columns() -> None:
    raw = pd.DataFrame(
        {
            "dataset_key": ["dataset_0172"],
            "target_name": ["wear rate"],
            "model_key": ["ridge"],
            "objective_value": [0.72],
        }
    )

    normalized = normalize_score_table(raw)

    assert normalized["score"].iloc[0] == 0.72
    assert normalized["variant"].iloc[0] == "original"
    assert normalized["output_mode"].iloc[0] == "single_output"
