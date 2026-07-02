# Preprocessing

Reusable preprocessing implementation lives under `src/friction_surrogate_xai/preprocessing/`.

Generate preprocessing artifacts with:

```bash
python -m friction_surrogate_xai.preprocessing
```

Saved pipeline artifacts are unfitted. Scaling and constant-feature detection must happen only when the sklearn pipeline is fit inside each cross-validation fold.
