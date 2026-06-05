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


## Validators for All Model Types

### DRM-only latent-codebook checkpoint

DRM-only can only evaluate samples that exist in the checkpoint latent codebook. If you use unseen `test_pairs`, samples not present in checkpoint `sample_names` are skipped.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 tools_2_metrics/validate_drm_only_checkpoints.py \
  --checkpoint /path/to/latest_net_DRMOnly.pth \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist train_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --batch_size 30 \
  --num_workers 2 \
  --tau 0.01 \
  --output_csv eval_drm_only.csv
```

### Image encoder + DRM checkpoint

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 tools_2_metrics/validate_image_sdf_checkpoints.py \
  --checkpoint /path/to/latest_net_DRMImage.pth \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist test_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --batch_size 30 \
  --num_workers 2 \
  --tau 0.01 \
  --output_csv eval_image_drm.csv
```

### MTM-z + image encoder + DRM checkpoint

Supports `add`, `replace`, and `concat` checkpoints. Fusion mode and scale are read from the checkpoint config.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 tools_2_metrics/validate_mtm_z_checkpoints.py \
  --checkpoint /path/to/latest_net_MTMZImageDRM.pth \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist test_pairs \
  --datamode aligned \
  --batch_size 16 \
  --num_workers 2 \
  --tau 0.01 \
  --output_csv eval_mtm_z.csv
```

### Combined comparison table

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 tools_2_metrics/validate_all_checkpoints.py \
  --drm_only_checkpoints /path/to/latest_net_DRMOnly.pth \
  --drm_only_names drm_only \
  --image_checkpoints /path/to/latest_net_DRMImage.pth \
  --image_names image_drm \
  --mtm_z_checkpoints /path/to/latest_net_MTMZImageDRM.pth \
  --mtm_z_names mtm_z \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist test_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --batch_size 16 \
  --num_workers 2 \
  --tau 0.01 \
  --output_csv eval_all_models.csv
```

## Mesh Metrics From Predicted OBJ vs GT OBJ

The validators can now compute mesh metrics by reconstructing a predicted mesh from the model during validation and comparing it to GT OBJ meshes.

GT OBJ location defaults to:

```text
<dataroot>/obj/<datalist>/<sample_name>.obj
```

This matches `tools/npz_to_obj_ball_pivoting.py`, which converts:

```text
<dataroot>/sdf/<split>/<sample_name>.npz
```

to:

```text
<dataroot>/obj/<split>/<sample_name>.obj
```

If the GT OBJ does not exist, validation will try to generate it automatically when `--auto_generate_gt_mesh 1`.

Example for one MTM-z checkpoint with mesh metrics on 10 samples:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 tools_2_metrics/validate_all_checkpoints.py \
  --mtm_z_checkpoints /path/to/latest_net_MTMZImageDRM.pth \
  --mtm_z_names mtm_z \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist test_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --batch_size 8 \
  --num_workers 2 \
  --tau 0.01 \
  --compute_mesh_metrics 1 \
  --mesh_eval_count 10 \
  --mesh_resolution 96 \
  --mesh_bounds_source sample \
  --num_surface_points 100000 \
  --output_csv eval_mtm_z_with_mesh.csv
```

For a quick smoke test, reduce the workload:

```bash
--mesh_eval_count 2 --mesh_resolution 64 --num_surface_points 20000
```

Mesh metrics reported:

```text
ChamferDistance ↓
FScore ↑
NormalConsistency ↑
```
