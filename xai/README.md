# XAI

Explainability workflows are implemented in `src/friction_surrogate_xai/xai/`.

The framework supports SHAP, permutation importance, tree feature importance,
tree-interpreter style local contributions, LIME, CSV reports, figures, Markdown
summaries, and MLflow artifact logging.

Run one report from the repository root:

```bash
python -m friction_surrogate_xai.xai --dataset dataset_0172 --target "wear rate" --model random_forest --no-mlflow
```

Generated artifacts are written to `reports/xai/` and ignored by Git.
