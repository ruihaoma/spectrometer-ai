# Final Model Artifacts

This directory contains the model selected from the completed 350-epoch training run.

## Files

- `best_model.pt`: selected PyTorch checkpoint from epoch 74.
- `checkpoint_metadata.json`: checkpoint hash, model dimensions, sample counts, and current project paths.
- `metrics.json`: complete metrics and loss history from the original training run.
- `model_summary.md`: concise settings recorded by the original training script.
- `loss_curve.png`: training and validation loss curves.

The expected SHA256 of `best_model.pt` is:

```text
dda1be29ae42f424d4ef000138c7e9de83d10cfbd4d968cef7304ea8f8b44ae0
```

## Path Note

`metrics.json` and `model_summary.md` preserve paths written by the original training run, including `data/processed/relative_calib_mixed_v1` and `outputs/train_runs/...`. These values are historical records and are not the current reproduction paths.

The current repository regenerates the omitted dataset under `data/generated/relative_calib_mixed_v1` and uses `configs/train/relative_calib_mixed_v1_80k_train.yaml`. The mapping between original and current paths is recorded in `checkpoint_metadata.json`.

Run `python verification/verify_project.py` to validate the checkpoint hash, architecture, training settings, metrics, and a CPU inference pass.
