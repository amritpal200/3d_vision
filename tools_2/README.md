# DRM-Only Bootstrap (No MTM)

This folder contains a minimal starting pipeline to train only an SDF DRM model and reconstruct coarse human meshes.

## 1) Train DRM-only baseline

```bash
python3 tools_2/train_drm_only.py \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist train_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --name DRM_only_bootstrap \
  --checkpoints_dir checkpoints \
  --num_epochs 20 \
  --batch_size 16
```

Notes:
- Uses precomputed SDF files from: `dataroot/sdf/<datalist>/*.npz`
- Learns one latent code per training sample (codebook) plus shared SDF MLP.

## 2) Reconstruct mesh from checkpoint

```bash
python3 tools_2/reconstruct_drm_only_mesh.py \
  --checkpoint checkpoints/aligned/DRM_only_bootstrap/best_net_DRMOnly.pth \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist train_pairs \
  --sample_index 0 \
  --resolution 96 \
  --output_obj mesh_results/drm_only_sample0.obj
```

If dataset loading for bounds fails, the script falls back to a fixed cube volume.
