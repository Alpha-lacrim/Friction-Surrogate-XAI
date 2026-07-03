# Models

Trained binary artifacts and model cards belong here. Binary artifacts should be
generated locally and are ignored by Git.

The model registry is implemented under `src/friction_surrogate_xai/models/`
and configured by `configs/modeling.yaml`. It defines conservative,
small-data-first defaults for every supported regressor:

- regularized linear models
- kernel models
- probabilistic Gaussian Process Regression
- shallow or regularized tree ensembles
- XGBoost and LightGBM with regularization and early-stopping policy
- shallow MLP only

Do not add high-capacity defaults without also adding explicit overfitting
controls and tests.

Hyperparameter tuning is implemented separately under
`src/friction_surrogate_xai/optimization/`. It uses Random Search for all
selected models, then Optuna only for the top 3 models from Stage 1. Grid Search
is intentionally not part of this project.
