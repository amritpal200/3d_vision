# MTM z Projection For Image-Conditioned DRM

This folder trains the new MTM latent projection head:

```text
person image -> frozen image encoder -> z_image
MTM agnostic + cloth -> frozen MTM features -> trainable z_proj -> z_mtm
z_image + z_mtm + 3D points -> frozen DRM -> SDF
```

Only `mtm.z_proj` is trainable. The MTM backbone, image encoder, and DRM are
frozen. There is no latent distillation loss; supervision comes from the SDF
loss through the frozen DRM.

## Train

```bash
python3 tools_2_mtm_z/train_mtm_z_for_image_drm.py \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist train_pairs \
  --datamode aligned \
  --image_drm_checkpoint /path/to/best_net_DRMImage.pth \
  --pretrained_mtm_checkpoint /path/to/latest_net_MTM.pth \
  --checkpoints_dir checkpoints \
  --name MTM_z_for_image_DRM \
  --batch_size 8 \
  --num_epochs 200
```


## Reconstruct

```bash
python3 tools_2_mtm_z/reconstruct_from_mtm_z.py \
  --checkpoint checkpoints/aligned/MTM_z_for_image_DRM/best_net_MTMZImageDRM.pth \
  --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
  --datalist test_pairs \
  --sample_index 0 \
  --output_obj mesh_results/mtm_z.obj \
  --resolution 96
```

