# Preprocessing Artifacts

Generated preprocessing pipeline artifacts are written here by:

```bash
python -m friction_surrogate_xai.preprocessing
```

Generated artifacts are ignored by Git and logged to MLflow. Saved pipeline objects are intentionally unfitted so scalers and constant-feature detection are fit only inside cross-validation folds.

