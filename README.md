# Friction Surrogate XAI

Production-oriented research skeleton for developing reliable, interpretable, and overfitting-resistant surrogate models for friction-processed composite properties from extremely small experimental datasets.

This repository currently contains project infrastructure, a complete configurable data layer, an automated research-grade EDA module, leakage-safe preprocessing pipeline builders, a reusable evaluation framework, an overfitting-first model audit layer, staged hyperparameter optimization, discrete-input dataset comparison workflows, and an explainability framework. It intentionally does not implement uncertainty quantification yet.

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
xai/                      Explainability workflow docs and helper scripts
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

These checks validate the skeleton, configs, importability, data layer, EDA module, preprocessing pipelines, evaluation framework, overfitting audit layer, staged hyperparameter optimization, discrete-input comparison workflow, and explainability framework. Tests that need raw Excel/PDF files are skipped when local raw files are absent.

## Future Work

The next implementation phase should add uncertainty quantification on top of the selected original/discrete models. Keep model validation wired so preprocessing is cloned and fit inside each CV fold before any model sees validation data.
