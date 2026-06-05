# SMPL Fitting From OBJ/PLY

This folder fits an SMPL body to an existing `.obj` or `.ply` reconstruction without training another model.

Pipeline:

```text
OBJ/PLY mesh or point cloud -> sample target points -> optimize SMPL -> save SMPL params + fitted SMPL OBJ
```

## Requirements

You need the official SMPL model file, for example:

```text
SMPL_NEUTRAL.pkl
SMPL_MALE.pkl
SMPL_FEMALE.pkl
```

Python packages:

```bash
pip install smplx trimesh scipy
```

## Example

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 tools_2_smpl/fit_smpl_to_mesh.py \
  --input_mesh mesh_results/unseen_person.obj \
  --smpl_model_path /path/to/SMPL_NEUTRAL.pkl \
  --gender neutral \
  --output_params smpl_results/unseen_person_smpl.npz \
  --output_pkl smpl_results/unseen_person_smpl.pkl \
  --output_obj smpl_results/unseen_person_smpl.obj \
  --num_target_points 20000 \
  --num_iters 1000 \
  --init_iters 200
```

For a `.ply` point cloud:

```bash
python3 tools_2_smpl/fit_smpl_to_mesh.py \
  --input_mesh mesh_results/unseen_person.ply \
  --smpl_model_path /path/to/SMPL_NEUTRAL.pkl \
  --output_params smpl_results/unseen_person_smpl.npz \
  --output_obj smpl_results/unseen_person_smpl.obj
```

## Outputs

The `.npz` contains:

```text
global_orient  (1, 3)
body_pose      (1, 69)
betas          (1, num_betas)
transl         (1, 3)
scale          (1,)
vertices       fitted SMPL vertices after scale/translation
faces          SMPL triangle faces
```

The `.obj` is the fitted SMPL body mesh for visual inspection.

## Notes

SMPL is a body model, not a clothing model. If your DRM mesh has loose clothes, hair, coats, or dresses, the fitted SMPL body will approximate the body underneath rather than reproduce the clothed surface.
