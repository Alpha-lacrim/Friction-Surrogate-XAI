# Optimization Artifacts

Generated hyperparameter optimization reports are written here.

The workflow is intentionally staged for tiny experimental datasets:

- Stage 1: Random Search for every enabled model.
- Stage 2: select the top 3 models from Stage 1.
- Stage 3: Optuna Bayesian optimization only for those top 3 models.

Generated run directories, tables, plots, and MLflow artifacts are ignored by Git.

Run example:

```bash
python experiments/run_hyperparameter_optimization.py --dataset dataset_0172 --target "wear rate" --models ridge elasticnet random_forest
```
