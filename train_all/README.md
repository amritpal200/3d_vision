# train_all

This folder contains a self-contained image-conditioned fusion pipeline for human mesh reconstruction.

## Files

- `fusion_sdf_model.py`: shared image encoder plus coarse+residual DRM fusion module.
- `train_mtm_drm_fusion.py`: trains the fusion model using MTM-produced latent `z` and image inputs.
- `reconstruct_mtm_drm_fusion.py`: reconstructs a single mesh from a saved fusion checkpoint.

## Training idea

The pipeline uses:

- MTM to produce latent `z` from the aligned image inputs.
- An additional image encoder over `person + agnostic` inputs.
- A DRM coarse branch initialized from a pretrained checkpoint.
- A residual DRM branch initialized from its own pretrained checkpoint and then trained on top of the fused latent.

The final SDF is:

`SDF = coarse(z_fused, p) + residual(z_fused, p)`

where `z_fused = z + proj(image_features)`.

## Example training command

```bash
cd /home/asingh/proves/3d/3d_vision
conda activate prova
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 train_all/train_mtm_drm_fusion.py \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist train_pairs \
  --datamode aligned \
  --mtm_ckpt /path/to/best_net_MTM.pth \
  --pretrained_coarse_checkpoint /path/to/best_net_DRM.pth \
  --pretrained_residual_checkpoint /path/to/best_net_DRMResidual.pth \
  --checkpoints_dir /data/113-1/users/asingh/project/3d/checkpoints/train_all \
  --name fusion_run \
  --batch_size 8 \
  --num_epochs 200 \
  --gpu_id 0
```

## Example reconstruction command

```bash
cd /home/asingh/proves/3d/3d_vision
conda activate prova
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 train_all/reconstruct_mtm_drm_fusion.py \
  --checkpoint /data/113-1/users/asingh/project/3d/checkpoints/train_all/aligned/fusion_run/best_net_train_all_fusion.pth \
  --mtm_ckpt /path/to/best_net_MTM.pth \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist test_pairs \
  --sample_index 0 \
  --output_obj mesh_results/fusion.obj \
  --gpu_id 0 \
  --save_point_cloud
```
