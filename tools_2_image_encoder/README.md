# Image-Conditioned DRM

This folder is isolated from the existing `tools_2` DRM-only pipeline.
It replaces the per-sample latent codebook with:

```text
image -> ConvImageEncoder -> latent z -> existing models_2.DRMSDFModel -> SDF -> mesh
```

The original `tools_2/train_drm_only.py` and `models_2/drm_only_model.py` are not modified.

## Files

- `image_encoder_model.py`: simple configurable convolutional encoder.
- `common.py`: dataset, checkpoint, image preprocessing, and SDF-loss helpers.
- `train_drm_image_conditioned.py`: image-conditioned DRM training.
- `reconstruct_from_image.py`: reconstruct from an image path.

## Training

Run from the `3d_vision` directory:

```bash
python3 tools_2_image_encoder/train_drm_image_conditioned.py \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist train_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --name DRM_image_conditioned \
  --checkpoints_dir checkpoints \
  --num_epochs 200 \
  --batch_size 16 \
  --latent_dim 128 \
  --sdf_hidden_dim 512 \
  --sdf_num_layers 8 \
  --pe_L 6
```

Resume from the latest checkpoint under the run directory:

```bash
python3 tools_2_image_encoder/train_drm_image_conditioned.py \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist train_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --name DRM_image_conditioned \
  --checkpoints_dir checkpoints \
  --resume
```

Checkpoints contain `encoder_state`, `drm_state`, and `config` plus optimizer
and progress metadata for resume.

## Reconstruction

```bash
python3 tools_2_image_encoder/reconstruct_from_image.py \
  --checkpoint checkpoints/aligned/DRM_image_conditioned/best_net_DRMImage.pth \
  --image_path /path/to/unseen_person.png \
  --output_obj mesh_results/unseen_person.obj \
  --resolution 96
```

By default reconstruction uses cube bounds `[-1.2, 1.2]^3`. For known samples,
you can pass `--bounds_npz /path/to/sample.npz` to infer bounds from SDF or
surface points, or pass `--x_bounds`, `--y_bounds`, and `--z_bounds` manually.

