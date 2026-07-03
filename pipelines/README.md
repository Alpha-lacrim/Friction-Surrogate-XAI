# Pipelines

Top-level pipeline notes and future shell wrappers belong here. The executable
Python orchestration code lives in `src/friction_surrogate_xai/pipelines/`.

Run the final resumable pipeline with:

```bash
python -m friction_surrogate_xai.pipelines
```

The final pipeline coordinates these project families:

1. Original datasets, single-output regression
2. Original datasets, multi-output regression
3. Discrete-input datasets, single-output regression with Top 3 models
4. Discrete-input datasets, multi-output regression with Top 3 models

Generated state, reports, and pipeline-level MLflow artifacts are local outputs
under `reports/final_pipeline/`.
