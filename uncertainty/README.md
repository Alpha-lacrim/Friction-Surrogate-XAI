# Uncertainty

Uncertainty workflows are implemented in `src/friction_surrogate_xai/uncertainty/`.

The framework supports:

- Gaussian Process predictive mean, variance, intervals, and confidence bands
- bootstrap out-of-bag prediction intervals for non-GPR models
- coverage probability and interval-width summaries
- model comparison reports
- MLflow artifact and metric logging

Run one report from the repository root:

```bash
python -m friction_surrogate_xai.uncertainty --dataset dataset_0172 --target "wear rate" --models gaussian_process_regression ridge --no-mlflow
```

Generated artifacts are written to `reports/uncertainty/` and ignored by Git.
