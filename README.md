# Image-Conditioned 3D Human Reconstruction and Virtual Try-On

This repository extends [M3D-VTON](https://github.com/fyviezhao/M3D-VTON)
with continuous 3D human reconstruction based on signed distance functions
(SDFs). It contains two independently trained branches:

1. **3D reconstruction:** an image, optionally enriched with an MTM feature,
   conditions a DRM SDF network that produces a human mesh or point cloud.
2. **2D virtual try-on:** the TFM model generates a dressed person image. The
   version in `TFM/` adds an SSIM loss to improve garment structure and texture.

The `final_reconstruction/` stage combines the geometry from the first branch
with the try-on image from the second branch to create a colored 3D point cloud.

```text
3D branch: image -> image encoder (+ optional MTM z) -> DRM -> SDF -> OBJ/PLY
2D branch: person + garment -> MTM warping -> TFM -> try-on PNG

Final reconstruction: geometry PLY + try-on PNG -> colored PLY
```

## Repository Structure

| Path | Purpose |
| --- | --- |
| `tools/precompute_sdf_dataset.py` | Builds SDF supervision from the MPV3D front and back depth maps. |
| `tools_2/` | DRM-only baseline with one trainable latent code per training sample. |
| `tools_2_image_encoder/` | Image-conditioned DRM training and reconstruction. |
| `tools_2_mtm_z/` | Image-DRM extended with a projected MTM latent vector. |
| `tools_2_metrics/` | Common evaluation for DRM-only, Image-DRM, and Image-DRM+MTM checkpoints. |
| `TFM/` | Separate SSIM-enhanced 2D try-on model implementation and documentation. |
| `final_reconstruction/` | Combines an SDF-generated point cloud with a TFM try-on image. |
| `original_M3D_VTON_files/` | Original generic M3D-VTON train, test, and RGB-D conversion entry points. |
| `models/`, `data/`, `options/`, `util/` | Shared M3D-VTON model, dataset, option, and utility code. |

Component-specific details are available in the README inside each folder.

## Installation

Run all commands from the repository root:

```bash
cd /path/to/3d_vision

conda create -n m3d-sdf python=3.12
conda activate m3d-sdf
pip install -r requirements.txt
pip install pytorch-msssim tensorboard
```

Install a PyTorch build compatible with the CUDA version on your machine if
the pinned build in `requirements.txt` is not suitable.

The final texturing stage can also be installed independently:

```bash
pip install -r final_reconstruction/requirements.txt
```

## Dataset

Download and preprocess the
[MPV3D dataset](https://drive.google.com/file/d/1qcynpXZ9eSlzTV-RDCr-Yip3GcuU314h/view?usp=sharing).
The main scripts expect a structure similar to:

```text
MPV3D/
├── train_pairs.txt
├── test_pairs.txt
├── image/
├── cloth/
├── cloth-mask/
├── image-parse/
├── pose/
├── palm-mask/
├── depth/
├── sdf/
│   ├── train_pairs/
│   └── test_pairs/
└── obj/
    └── test_pairs/
```

Generate SDF supervision for every split used during training or evaluation:

```bash
python3 tools/precompute_sdf_dataset.py \
  --dataroot /path/to/MPV3D \
  --split train_pairs \
  --num_points 10000 \
  --sigma 0.01

python3 tools/precompute_sdf_dataset.py \
  --dataroot /path/to/MPV3D \
  --split test_pairs \
  --num_points 10000 \
  --sigma 0.01
```

The files are written to `MPV3D/sdf/<split>/`. Missing dataset items are
reported and skipped by the safe dataset wrappers used in the new pipelines.

## Pretrained Models

Download the project checkpoints from the following placeholder link:

**[Download pretrained models from Google Drive](https://drive.google.com/drive/folders/169KMyA8Z5wpeanLZioBYCSHSPxVx-rxw?usp=sharing)**

```text
pretrained/
├── latest_net_DRMOnly.pth
├── latest_net_DRMImage.pth
├── latest_net_MTMZImageDRM.pth
├── latest_net_MTM.pth
└── latest_net_TFM.pth
```

The new reconstruction scripts read their architecture settings from the
checkpoint configuration, so architecture arguments normally do not need to
be repeated at inference time.

## 3D Reconstruction

### 1. DRM-Only Baseline

This baseline learns a latent codebook indexed by training-sample ID. It is
useful as a DRM pretraining stage but cannot reconstruct an unseen image
without a corresponding codebook entry.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python3 tools_2/train_drm_only.py \
  --dataroot /path/to/MPV3D \
  --datalist train_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --name DRM_only \
  --checkpoints_dir checkpoints \
  --num_epochs 200 \
  --batch_size 16
```

### 2. Image-Conditioned DRM

The image encoder replaces the sample-index codebook:

```text
image -> encoder -> z_image
(z_image, positional_encoding(query_point)) -> DRM -> predicted SDF
```

Train the image encoder and DRM jointly:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python3 tools_2_image_encoder/train_drm_image_conditioned.py \
  --dataroot /path/to/MPV3D \
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
  --pe_L 6 \
  --lambda_coarse 2.0 \
  --lambda_surface 0.1 \
  --lambda_sign 0.1
```

To initialize from a DRM-only checkpoint, add:

```text
--pretrained_drm_checkpoint /path/to/latest_net_DRMOnly.pth
--auto_drm_arch_from_checkpoint 1
--freeze_drm 0
```

Set `--freeze_drm 1` to train only the image encoder. Use `--resume` or
`--resume_checkpoint /path/to/checkpoint.pth` to continue an interrupted run.

Reconstruct an unseen image:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python3 tools_2_image_encoder/reconstruct_from_image.py \
  --checkpoint /path/to/latest_net_DRMImage.pth \
  --image_path /path/to/person.png \
  --output_obj mesh_results/person.obj \
  --output_point_cloud mesh_results/person.ply \
  --resolution 96 \
  --bounds_preset human
```

Reconstruct a random subset from a folder:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python3 tools_2_image_encoder/reconstruct_from_image.py \
  --checkpoint /path/to/latest_net_DRMImage.pth \
  --image_dir /path/to/images \
  --num_images 10 \
  --output_dir mesh_results/image_batch \
  --resolution 96
```

Batch outputs are placed in `mesh_results/image_batch/obj/` and
`mesh_results/image_batch/ply/`.

### 3. Image-DRM with MTM Latent

This model concatenates the image latent and the projected MTM latent before
passing them to the DRM. The MTM model also requires the person, garment, and
agnostic inputs associated with each dataset pair.

Train only the new MTM projection while freezing MTM, the image encoder, and
the DRM:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python3 tools_2_mtm_z/train_mtm_z_for_image_drm.py \
  --dataroot /path/to/MPV3D \
  --datalist train_pairs \
  --datamode aligned \
  --image_drm_checkpoint /path/to/latest_net_DRMImage.pth \
  --pretrained_mtm_checkpoint /path/to/latest_net_MTM.pth \
  --checkpoints_dir checkpoints \
  --name MTM_z_concat_projection \
  --batch_size 16 \
  --num_epochs 200 \
  --mtm_fusion_mode concat \
  --mtm_z_scale 1.0 \
  --train_mode mtm_z_only \
  --zero_init_z_proj 0 \
  --lr_z_proj 3e-4 \
  --lambda_coarse 2.0 \
  --lambda_surface 0.1 \
  --lambda_sign 0.1
```

For joint fine-tuning, replace the training options with:

```text
--train_mode all --lr_mtm 1e-5 --lr_encoder 1e-6 --lr_drm 1e-6
```

Reconstruct by image path:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python3 tools_2_mtm_z/reconstruct_from_mtm_z.py \
  --checkpoint /path/to/latest_net_MTMZImageDRM.pth \
  --dataroot /path/to/MPV3D \
  --datalist test_pairs \
  --datamode aligned \
  --image_path /path/to/MPV3D/image/person_whole_front.png \
  --output_obj mesh_results/mtm_z_person.obj \
  --output_point_cloud mesh_results/mtm_z_person.ply \
  --resolution 96 \
  --bounds_preset human
```

The image filename must be present in the selected datalist because MTM needs
the matching cloth and agnostic inputs.

## Evaluation

`tools_2_metrics/validate_all_checkpoints.py` compares any combination of the
three 3D model families. It reports Near-Surface SDF MAE, sign accuracy,
Chamfer distance, and F-score. Only one checkpoint group is required.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python3 tools_2_metrics/validate_all_checkpoints.py \
  --image_checkpoints /path/to/latest_net_DRMImage.pth \
  --image_names image_drm \
  --dataroot /path/to/MPV3D \
  --datalist test_pairs \
  --datamode aligned \
  --dataset_model MTM \
  --batch_size 16 \
  --num_workers 2 \
  --tau 0.01 \
  --compute_mesh_metrics 1 \
  --mesh_eval_count 10 \
  --mesh_resolution 96 \
  --mesh_bounds_source sample \
  --num_surface_points 20000 \
  --output_csv evaluation.csv
```

Use `--drm_only_checkpoints` or `--mtm_z_checkpoints` for the other model
families. Multiple groups may be supplied in one command. Mesh evaluation is
considerably more expensive than pointwise SDF evaluation; set
`--mesh_eval_count -1` only when evaluating the complete test set.

## 2D TFM with SSIM

`TFM/` is a separate 2D try-on experiment. It does not train the SDF models.
Its `TFM_model.py` adds SSIM to the original L1, VGG, and mask losses.

The generic M3D-VTON runner loads `models/TFM_model.py`, so activate the SSIM
implementation before training:

```bash
cp models/TFM_model.py models/TFM_model_original.py
cp TFM/TFM_model.py models/TFM_model.py
```

First generate the MTM warping results used as `--warproot` for both splits:

```bash
PYTHONPATH=. python3 original_M3D_VTON_files/test.py \
  --model MTM \
  --name MTM \
  --dataroot /path/to/MPV3D \
  --datalist train_pairs \
  --checkpoints_dir checkpoints \
  --results_dir results

PYTHONPATH=. python3 original_M3D_VTON_files/test.py \
  --model MTM \
  --name MTM \
  --dataroot /path/to/MPV3D \
  --datalist test_pairs \
  --checkpoints_dir checkpoints \
  --results_dir results_test
```

Train and test the SSIM-enhanced TFM:

```bash
PYTHONPATH=. python3 original_M3D_VTON_files/train.py \
  --model TFM \
  --name TFM_ssim \
  --dataroot /path/to/MPV3D \
  --warproot results/aligned/MTM/train_pairs \
  --datalist train_pairs \
  --checkpoints_dir checkpoints

PYTHONPATH=. python3 original_M3D_VTON_files/test.py \
  --model TFM \
  --name TFM_ssim \
  --dataroot /path/to/MPV3D \
  --warproot results_test/aligned/MTM/test_pairs \
  --datalist test_pairs \
  --checkpoints_dir checkpoints \
  --results_dir results_ssim
```

See [`TFM/README.md`](TFM/README.md) for the loss definition and reported 2D
try-on results.

## Final Reconstruction

`final_reconstruction/` is a post-processing stage, not another trained
network. It combines:

- a geometry-only `.ply` produced by the Image-DRM or Image-DRM+MTM model; and
- a 2D try-on `.png` produced by the TFM branch.

The current script processes ten paired subjects. Place matching files as:

```text
final_reconstruction/
├── 10modelos3d/
│   ├── 1.ply
│   └── ... 10.ply
└── 10imagenes2d/
    ├── baseline/1.png ... 10.png
    ├── ssim_100/1.png ... 10.png
    └── ssim_200/1.png ... 10.png
```

The number identifies the subject, so `N.ply` must correspond to `N.png`.
Generate the colored point clouds from inside the folder:

```bash
cd final_reconstruction

python3 texture_10.py --method ssim_200
python3 texture_10.py --method baseline
python3 texture_10.py --method ssim_100
```

Outputs are saved under `resultados/10_<method>/`. Visualize one result or all
ten:

```bash
python3 view_3d.py resultados/10_ssim_200/recon_1_ssim_200_pcd.ply
python3 ver_10.py --metodo ssim_200
```

See [`final_reconstruction/README.md`](final_reconstruction/README.md) for the
projection, front/back coloring, alignment, and cleanup details.

## Notes

- Reconstruction resolution controls the cost and detail of Marching Cubes;
  `96` is the standard setting used in this project.
- Keep checkpoint configuration metadata intact. The image and MTM-z
  reconstruction scripts use it to rebuild the correct model architecture.
- Use human or sample-derived bounds and keep the largest connected component
  to suppress disconnected SDF surfaces.
- `final_reconstruction/` expects point clouds, whereas the reconstruction
  scripts can save both `.obj` meshes and `.ply` point clouds.

## License and Attribution

The MPV3D dataset and original M3D-VTON code are restricted to non-commercial
research and educational use. Please also follow the licenses of all included
dependencies and pretrained models.

```bibtex
@InProceedings{M3D-VTON,
    author    = {Zhao, Fuwei and Xie, Zhenyu and Kampffmeyer, Michael and
                 Dong, Haoye and Han, Songfang and Zheng, Tianxiang and
                 Zhang, Tao and Liang, Xiaodan},
    title     = {M3D-VTON: A Monocular-to-3D Virtual Try-On Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on
                 Computer Vision (ICCV)},
    year      = {2021},
    pages     = {13239--13249}
}
```
