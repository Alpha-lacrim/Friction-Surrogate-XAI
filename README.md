# Friction Surrogate XAI

Production-oriented research skeleton for developing reliable, interpretable, and overfitting-resistant surrogate models for friction-processed composite properties from extremely small experimental datasets.

This repository currently contains project infrastructure, a complete configurable data layer, an automated research-grade EDA module, leakage-safe preprocessing pipeline builders, a reusable evaluation framework, an overfitting-first model audit layer, staged hyperparameter optimization, discrete-input dataset comparison workflows, an explainability framework, uncertainty estimation workflows, statistical model comparison, and a resumable final orchestration pipeline.

## Research Scope

The project specification requires a small-data ML framework focused on scientific reliability rather than raw accuracy:

- three original Excel datasets with 36, 72, and 36 samples
- leakage-safe preprocessing through `sklearn.pipeline`
- single-output and multi-output regression
- original inputs and discretized-input variants
- repeated random seeds for sensitivity analysis
- complete MLflow tracking of experiments, metrics, parameters, and artifacts
- anti-overfitting evaluation using train/test gaps and fold stability
- uncertainty analysis with GPR prediction intervals and bootstrap intervals
- global and local interpretability with SHAP, permutation importance, and LIME
- final mechanical-engineering interpretation of surrogate model behavior

## Repository Layout

```text
configs/                  Project, dataset, MLflow, seed, and pipeline configs
data/raw/                 Canonical assignment PDF and Excel datasets
data/interim/             Future intermediate data artifacts
data/processed/           Future processed and discretized datasets
experiments/              Future experiment entrypoints and run scripts
pipelines/                Top-level pipeline docs and future shell wrappers
models/                   Future trained model artifacts and model docs
preprocessing/            Future preprocessing scripts
evaluation/               Evaluation reports and statistical comparison docs
visualization/            Future plotting scripts and helpers
xai/                      Explainability workflow docs and helper scripts
uncertainty/              Uncertainty estimation workflow docs
utils/                    Future top-level utility scripts
notebooks/                Future EDA and analysis notebooks
reports/                  Future figures, tables, screenshots, and final report
src/friction_surrogate_xai/ Package code for reusable infrastructure
tests/                    Skeleton tests
```

## Data

The three canonical raw Excel datasets are versioned under `data/raw/` for reproducibility. The assignment PDF remains local-only and ignored by Git.

| Dataset | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `Dataset 0136.xlsx` | 36 | 12 | Includes common targets plus `Temperature (°C)` and `Strain` |
| `Dataset 0172.xlsx` | 72 | 10 | Includes common targets only in the inspected file |
| `Dataset 3772.xlsx` | 36 | 10 | Includes common targets only |

The input features are:

- `Tool Shape`
- `Rotational Speed`
- `Plunging Speed`
- `Composite Volume Fraction (%)`

The `No.` column is sample metadata, not a predictive feature.

## Data Layer

The package exposes a configurable data layer under `friction_surrogate_xai.data`.

```python
from friction_surrogate_xai.data import DataLoader

datasets = DataLoader().load_all()
dataset_0136 = datasets["dataset_0136"]
print(dataset_0136.dataframe.shape)
print(dataset_0136.report.constants.constant_feature_columns)
```

The loader reads paths and schema rules from `configs/datasets.yaml` and generates:

- schema and shape validation
- datatype validation
- duplicate-row and duplicate-ID reports
- constant-column and constant-feature reports
- missing-value counts and ratios
- numeric descriptive statistics
- generated dataset metadata

Current real-data detection:

- `dataset_0136`: `Composite Volume Fraction (%)` is constant at `0`
- `dataset_0172`: `Composite Volume Fraction (%)` varies across `0` and `1`
- `dataset_3772`: `Composite Volume Fraction (%)` is constant at `1`

## EDA

Run automated EDA for every configured dataset:

```bash
python -m friction_surrogate_xai.eda
```

This generates local artifacts under `reports/eda/` and logs each dataset run to the MLflow experiment `friction-surrogate-xai-eda`.

Generated outputs include:

- publication-quality histograms, KDE plots, QQ plots, box plots, pair plots, and correlation heatmaps
- Pearson, Spearman, and Kendall correlation CSV matrices
- Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov normality tests
- confidence intervals, skewness, kurtosis, variance, and descriptive statistics
- IQR, Isolation Forest, and Local Outlier Factor outlier reports
- markdown summaries and CSV tables

Outliers are never removed by the EDA module. They are detected and reported only.

## Preprocessing

Generate unfitted preprocessing pipeline artifacts for every configured dataset:

```bash
python -m friction_surrogate_xai.preprocessing
```

The preprocessing system is configured by `configs/preprocessing.yaml` and builds sklearn `Pipeline` objects with:

- feature validation
- constant-feature removal
- `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or no scaling
- optional `OneHotEncoder`

Leakage policy:

- saved preprocessing pipelines are intentionally unfitted
- constant-feature removal learns constants only during `Pipeline.fit`
- scalers are inside the sklearn pipeline and must be fit inside CV folds
- never fit preprocessing on the full dataset before validation

Generated artifacts are written under `reports/preprocessing_artifacts/` and logged to MLflow experiment `friction-surrogate-xai-preprocessing`.

## Evaluation

The package exposes a reusable evaluation framework under `friction_surrogate_xai.evaluation`.

```python
from friction_surrogate_xai.evaluation import EvaluationReportGenerator

artifacts = EvaluationReportGenerator().generate(
    dataset_key="dataset_0172",
    model_name="future_model_name",
    y_train_true=y_train,
    y_train_pred=train_predictions,
    y_test_true=y_test,
    y_test_pred=test_predictions,
    target_names=("wear rate",),
    fold_metrics=cv_fold_metrics,
)
```

The framework is configured by `configs/evaluation.yaml` and generates:

- R2, RMSE, NRMSE, and MAE
- train/test gap tables
- fold-stability summaries with mean, standard deviation, and confidence intervals
- prediction-vs-actual plots
- residual plots
- optional sklearn learning curves and validation curves
- CSV reports and publication-ready Markdown tables

Generated artifacts are written under `reports/evaluation/` and logged to MLflow experiment `friction-surrogate-xai-evaluation` when logging is enabled.

## Overfitting Audits

The modeling layer is configured by `configs/modeling.yaml` and builds conservative estimator defaults for:

- Linear Regression, Ridge, and ElasticNet
- SVR with RBF kernel
- Gaussian Process Regression
- Random Forest, Extra Trees, and Gradient Boosting
- XGBoost and LightGBM
- shallow `MLPRegressor`

Anti-overfitting controls include regularization, shallow tree depth, minimum samples per leaf/split, row and feature subsampling, bootstrap aggregation, early stopping where compatible, five repeated random seeds, repeated KFold, LOOCV support, nested CV split infrastructure, and bootstrap out-of-bag evaluation.

Run an audit for a dataset target:

```bash
python experiments/run_overfitting_audit.py --dataset dataset_0172 --target "wear rate" --models ridge random_forest --no-mlflow
```

Generated reports compare training score, validation score, generalization gap, and variance across folds. Models crossing configured thresholds are flagged in `reports/evaluation/overfitting/`.

## Hyperparameter Optimization

The optimization layer is configured by `configs/optimization.yaml` and implemented under `friction_surrogate_xai.optimization`.

The workflow is deliberately staged:

- Stage 1: Random Search for every selected model.
- Stage 2: select the top 3 models by validation score.
- Stage 3: run Optuna Bayesian optimization only for those top 3 models.
- Grid Search is not used.

Run optimization for one dataset target:

```bash
python experiments/run_hyperparameter_optimization.py --dataset dataset_0172 --target "wear rate" --models ridge elasticnet random_forest --no-mlflow
```

Generated artifacts are written under `reports/optimization/`:

- best parameters as CSV and JSON
- full optimization history
- Stage 1 Random Search history
- Stage 2 top-model table
- Stage 3 Optuna history
- Optuna parameter importance tables
- optimization history plots, top-model plots, and parameter-importance plots

When MLflow is enabled, every Random Search and Optuna trial is logged as its own MLflow run.

## Discrete Inputs

The discretization layer is configured by `configs/discretization.yaml` and implemented under `friction_surrogate_xai.discretization`.

It generates one discrete-input dataset per original dataset:

- `dataset_0136_discrete`
- `dataset_0172_discrete`
- `dataset_3772_discrete`

Configured continuous process inputs are converted to integer bins while target columns are preserved unchanged. The binning method is configurable, with quantile and uniform binning supported. Constant and low-cardinality integer features are handled explicitly so no rows are removed.

Run the full workflow:

```bash
python -m friction_surrogate_xai.discretization
```

Useful variants:

```bash
python -m friction_surrogate_xai.discretization --skip-comparison --no-mlflow
python -m friction_surrogate_xai.discretization --targets "wear rate" --models ridge elasticnet linear_regression
```

Generated outputs include:

- processed discrete datasets under `data/processed/discrete/`
- discretization metadata and bin mappings under `reports/discretization/`
- original-vs-discrete comparison reports for the configured targets
- MLflow logs for dataset generation and comparison runs

For comparisons, the workflow resolves the Top 3 models from existing optimization artifacts when available, otherwise it uses the configured fallback Top 3 for fresh clones. Original and discrete variants use identical models, hyperparameters, CV splits, seeds, metrics, and fold-local preprocessing.

## Explainability

The explainability layer is configured by `configs/xai.yaml` and implemented under `friction_surrogate_xai.xai`.

Run one XAI report:

```bash
python -m friction_surrogate_xai.xai --dataset dataset_0172 --target "wear rate" --model random_forest --no-mlflow
```

Generated artifacts are written under `reports/xai/` and include:

- global and local SHAP tables
- SHAP beeswarm, summary, waterfall, dependence, and interaction plots
- permutation-importance tables and figures
- tree feature-importance tables and figures when the estimator exposes them
- tree-interpreter style local contribution reports, with Tree SHAP fallback when `treeinterpreter` is unavailable
- LIME local explanation tables and figures
- Markdown scientific interpretations covering important variables, positive and negative effects, nonlinear behavior, feature interactions, and possible engineering interpretation

When MLflow is enabled, the complete XAI run directory is logged to experiment `friction-surrogate-xai-explainability`.

## Uncertainty

The uncertainty layer is configured by `configs/uncertainty.yaml` and implemented under `friction_surrogate_xai.uncertainty`.

Run one uncertainty report:

```bash
python -m friction_surrogate_xai.uncertainty --dataset dataset_0172 --target "wear rate" --models gaussian_process_regression ridge random_forest --no-mlflow
```

Generated artifacts are written under `reports/uncertainty/` and include:

- GPR predictive mean, predictive variance, prediction intervals, and confidence-band plots
- bootstrap out-of-bag prediction intervals for non-GPR models
- coverage probability, interval width, and predictive-variance summaries
- model comparison reports ranking interval calibration and width
- Markdown summaries and MLflow artifact logging

All uncertainty estimators fit preprocessing inside the GPR CV folds or bootstrap samples before predicting held-out samples.

## Statistical Comparison

The statistical-comparison layer is configured by `configs/statistical_comparison.yaml` and implemented under `friction_surrogate_xai.statistical_comparison`.

Run the auto-discovery workflow:

```bash
python -m friction_surrogate_xai.statistical_comparison --no-mlflow
```

Or pass explicit normalized score CSV files:

```bash
python -m friction_surrogate_xai.statistical_comparison --input path/to/scores.csv --no-mlflow
```

Generated artifacts are written under `reports/statistical_comparison/` and include:

- normalized score tables
- Wilcoxon signed-rank pairwise tests
- Friedman omnibus tests
- Nemenyi post-hoc pairwise comparisons
- average-rank tables
- significant-finding tables
- p-value heatmaps and average-rank plots
- Markdown summaries and MLflow artifact logging

The workflow automatically supports the project comparison families when matching score rows are available: top models, original-vs-discrete inputs, and single-output-vs-multi-output runs.

## Final Pipeline

The final orchestration layer is configured by `configs/final_pipeline.yaml` and implemented under `friction_surrogate_xai.pipelines`.

Run the complete resumable workflow:

```bash
python -m friction_surrogate_xai.pipelines
```

For a dry local run without MLflow logging:

```bash
python -m friction_surrogate_xai.pipelines --no-mlflow
```

Fast smoke run for checking the full orchestration wiring on one dataset, one target, and one model:

```bash
python -m friction_surrogate_xai.pipelines --config configs/final_pipeline_smoke.yaml --no-mlflow
```

Use the dedicated `friction-xai` environment for local runs. The conda `base` environment on this machine is not recommended because it can mix NumPy 2.x with older compiled packages.

The final pipeline coordinates:

- dataset loading and validation
- leakage-safe preprocessing artifact generation
- EDA generation
- discrete-input dataset generation and original-vs-discrete comparison
- single-output and multi-output all-model training/evaluation audits
- single-output and multi-output staged Random Search and Optuna optimization
- uncertainty reports
- XAI reports
- statistical comparison
- final Markdown/CSV project report generation

The pipeline writes persistent state after every stage under `reports/final_pipeline/<run_id>/state/pipeline_state.json`. With `resume: true`, completed stages are skipped on the next run, and with `continue_on_error: true`, later stages can still run after a failure so earlier experiments and artifacts are preserved. Final reports are written under `reports/final_pipeline/<run_id>/markdown/`.

### Latest Production Run

A full production run using `configs/final_pipeline.yaml` was completed locally with the default `run_id: latest`. All configured stages finished successfully:

- loaded and validated all 3 datasets
- generated preprocessing and EDA artifacts for all 3 datasets
- generated 3 discretized datasets and 17 original-vs-discrete comparisons
- ran 20 training/evaluation audit jobs
- ran 20 staged optimization jobs
- generated 17 uncertainty reports
- generated 51 XAI model-target reports
- generated statistical comparisons from 10,685 normalized score rows
- wrote the final project report

Inspect the latest production artifacts at:

- `reports/final_pipeline/latest/markdown/final_project_report.md`
- `reports/final_pipeline/latest/tables/pipeline_stage_status.csv`
- `reports/final_pipeline/latest/state/pipeline_state.json`

Component outputs are written under `reports/eda/`, `reports/preprocessing_artifacts/`, `reports/discretization/`, `reports/evaluation/overfitting/`, `reports/optimization/`, `reports/uncertainty/`, `reports/xai/`, and `reports/statistical_comparison/`. MLflow runs are written under `mlruns/` when MLflow logging is enabled.

Generated run artifacts are intentionally ignored by Git. Re-run the final pipeline locally to regenerate them. Because `resume: true` is enabled in the production config, running the same command again with `run_id: latest` will reuse completed stages unless `force_rerun: true` is set or the run id is changed.

For tiny datasets, Gaussian Process Regression may emit `ConvergenceWarning` messages when the fitted noise level reaches a configured kernel bound. These warnings are expected during production runs and do not indicate a failed stage.

### Production Evaluation Snapshot

The README keeps only headline evaluation evidence. Full per-target metrics, plots, and statistical tables are generated artifacts and should be inspected under `reports/` after each run.

Current `latest` run highlights:

- Stage 2 optimization selected the same Top 3 model families across all 20 optimization jobs: `gaussian_process_regression`, `extra_trees`, and `elasticnet`.
- Overfitting audits generated 1,628 model/target/metric risk records with training score, validation score, generalization gap, fold variance, `risk_level`, and `likely_overfitting` flags.
- Uncertainty evaluation generated 34 interval-calibration rows: GPR predictive intervals and bootstrap out-of-bag intervals for 17 dataset-target runs.
- Statistical comparison produced 2,813 significant pairwise findings from Wilcoxon signed-rank and Nemenyi post-hoc outputs.

Representative `dataset_0172` / `wear rate` result from the current run:

| Evaluation | Result |
| --- | --- |
| Optimization Top 3 | GPR validation R2 `0.974`, ElasticNet validation R2 `0.961`, Extra Trees validation R2 `0.897` |
| Overfitting audit | GPR and ElasticNet were low-risk; Extra Trees was flagged medium-risk because validation variance crossed the configured threshold |
| Uncertainty | At the 95% interval level, GPR coverage was `0.847` and Ridge bootstrap coverage was `0.417`; both under-covered and should be interpreted cautiously |

Key detailed tables:

- `reports/evaluation/overfitting/<dataset>/<target>/tables/overfitting_summary.csv`
- `reports/optimization/<dataset>/<target>/tables/stage2_top_models.csv`
- `reports/optimization/<dataset>/<target>/tables/best_parameters.csv`
- `reports/uncertainty/<dataset>/<target>/tables/comparison_report.csv`
- `reports/statistical_comparison/tables/significant_findings.csv`

## Environment

Recommended setup:

```bash
conda env create -f environment.yml
conda activate friction-xai
python -m pip install -e . --no-deps
```

For an already-created local environment, activate it first:

```powershell
conda activate friction-xai
python -m pip install -e . --no-deps
```

## Verification

Run:

```bash
python -m pytest
python -m friction_surrogate_xai
```

These checks validate the skeleton, configs, importability, data layer, EDA module, preprocessing pipelines, evaluation framework, overfitting audit layer, staged hyperparameter optimization, discrete-input comparison workflow, explainability framework, uncertainty estimation, statistical comparison, and final orchestration pipeline. Tests that need raw Excel/PDF files are skipped when local raw files are absent.
