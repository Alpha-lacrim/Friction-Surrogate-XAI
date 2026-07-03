# Evaluation

This directory documents the reusable evaluation layer implemented under
`src/friction_surrogate_xai/evaluation/`.

The framework accepts predictions and optional fold metrics from future model
pipelines, then generates:

- R2, RMSE, NRMSE, and MAE
- train/test gap reports
- fold-stability summaries with mean, standard deviation, and confidence intervals
- prediction-vs-actual plots
- residual plots
- optional sklearn learning curves
- optional sklearn validation curves
- CSV and Markdown reports
- MLflow metrics and artifacts

Generated evaluation artifacts are written to `reports/evaluation/` and ignored
by Git.
