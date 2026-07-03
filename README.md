# Friction Surrogate XAI

Production-oriented research skeleton for developing reliable, interpretable, and overfitting-resistant surrogate models for friction-processed composite properties from extremely small experimental datasets.

This repository currently contains project infrastructure, a complete configurable data layer, an automated research-grade EDA module, leakage-safe preprocessing pipeline builders, and a reusable evaluation framework. It intentionally does not implement model training, hyperparameter optimization, SHAP/LIME analysis, or uncertainty quantification yet.

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
pipelines/                Future top-level pipeline scripts
models/                   Future trained model artifacts and model docs
preprocessing/            Future preprocessing scripts
evaluation/               Evaluation reports and statistical comparison docs
visualization/            Future plotting scripts and helpers
xai/                      Future SHAP, LIME, and feature-importance workflows
uncertainty/              Future prediction interval and bootstrap workflows
utils/                    Future top-level utility scripts
notebooks/                Future EDA and analysis notebooks
reports/                  Future figures, tables, screenshots, and final report
src/friction_surrogate_xai/ Package code for reusable infrastructure
tests/                    Skeleton tests
```

## Data

Raw assignment files are expected locally in `data/raw/`, but they are ignored by Git. Keep the filenames below when placing the files in a fresh clone.

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

## Environment

Recommended setup:

```bash
conda env create -f environment.yml
conda activate friction-surrogate-xai
python -m pip install -e .
```

For local verification in the current workspace, the existing conda `base` environment was used because it already includes several required packages.

## Verification

Run:

```bash
python -m pytest
python -m friction_surrogate_xai
```

These checks validate the skeleton, configs, importability, data layer, EDA module, preprocessing pipelines, and evaluation framework. Tests that need raw Excel/PDF files are skipped when local raw files are absent. They do not train project models.

## Future Work

The next implementation phase should add model/pipeline factories with MLflow tracking. Keep model validation wired so the saved preprocessing pipeline is cloned and fit inside each CV fold, then call the reusable evaluation framework with the resulting predictions and fold metrics.
