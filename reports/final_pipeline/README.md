# Final Pipeline Artifacts

Generated orchestration state files, final project reports, and pipeline-level
MLflow artifacts are written here.

The final pipeline is resumable: completed stage status is persisted after each
stage so later failures do not remove earlier experiment artifacts.

Run it with:

```bash
python -m friction_surrogate_xai.pipelines
```
