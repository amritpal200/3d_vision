# SDF Metrics And Model Comparison

This folder provides reusable validation metrics for SDF-based reconstruction.

Primary metric:

```text
NearSurfaceMAE, using points where |gt_sdf| < tau
```

Default settings:

```text
tau = 0.01
delta = 0.01
num_surface_points = 100000
```

## Metrics

- `near_surface_mae`: lower is better.
- `sign_accuracy`: higher is better.
- `chamfer_distance`: lower is better.
- `f_score`: higher is better.
- `normal_consistency`: higher is better.

Mesh metrics need dense SDF grids. If you only have unordered point samples,
`near_surface_mae` and `sign_accuracy` are computed, while mesh metrics are
reported as `nan`.

## Reusable Function

```python
from tools_2_metrics import evaluate_sdf

metrics = evaluate_sdf(
    pred_sdf,
    gt_sdf,
    tau=0.01,
    delta=0.01,
)
```

For mesh metrics:

```python
metrics = evaluate_sdf(
    pred_sdf_grid.reshape(-1),
    gt_sdf_grid.reshape(-1),
    grid_shape=(96, 96, 96),  # nz, ny, nx
    bounds=((-0.8, 0.8), (-1.15, 1.15), (-0.8, 0.8)),
)
```

## Evaluate Saved Predictions

```bash
python3 tools_2_metrics/evaluate_sdf_predictions.py \
  --inputs model_a_val.npz model_b_val.npz \
  --model_names model_a model_b \
  --pred_key pred_sdf \
  --gt_key gt_sdf \
  --output_csv metrics.csv
```

With mesh metrics:

```bash
python3 tools_2_metrics/evaluate_sdf_predictions.py \
  --inputs model_a_grid.npz model_b_grid.npz \
  --model_names model_a model_b \
  --grid_shape 96 96 96 \
  --bounds -0.8 0.8 -1.15 1.15 -0.8 0.8 \
  --output_csv metrics_grid.csv
```

## Validate Image-Conditioned SDF Checkpoints

```bash
python3 tools_2_metrics/validate_image_sdf_checkpoints.py \
  --checkpoints /path/to/model_a.pth /path/to/model_b.pth \
  --model_names model_a model_b \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist val_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --batch_size 4 \
  --output_csv checkpoint_metrics.csv \
  --wandb_project m3d_drm \
  --wandb_mode online
```

This computes point metrics on validation SDF samples. Mesh metrics require
dense-grid SDF values, so they are skipped unless `--grid_shape nz ny nx` is
provided and each sample contains SDF values on that grid.


## Best Checkpoint Priority

Use this order:

1. `near_surface_mae` lower
2. `sign_accuracy` higher
3. `chamfer_distance` lower
4. `f_score` higher
5. `normal_consistency` higher

