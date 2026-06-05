
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 tools_2_metrics/validate_all_checkpoints.py \
#   --drm_only_checkpoints /path/to/latest_net_DRMOnly.pth \
#   --drm_only_names drm_only \
#   --image_checkpoints /path/to/latest_net_DRMImage.pth \
#   --image_names image_drm \
#   --mtm_z_checkpoints /path/to/latest_net_MTMZImageDRM.pth \
#   --mtm_z_names mtm_z \
#   --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
#   --datalist test_pairs \
#   --datamode aligned \
#   --dataset_model MTM \
#   --batch_size 16 \
#   --num_workers 2 \
#   --tau 0.01 \
#   --output_csv eval_all_models.csv



#!/usr/bin/env python3
"""Compare DRM-only, image-conditioned DRM, and MTM-z DRM checkpoints."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
IMAGE_ENCODER_DIR = os.path.join(PROJECT_ROOT, "tools_2_image_encoder")
for path in (PROJECT_ROOT, CURRENT_DIR, IMAGE_ENCODER_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from tools_2_image_encoder.common import create_image_sdf_dataset, get_device, safe_collate  # noqa: E402
from tools_2_mtm_z.common import create_mtm_dataset  # noqa: E402
from validation_common import add_mesh_eval_args, init_wandb, print_and_save_results  # noqa: E402
from validate_drm_only_checkpoints import build_dataset_opt, SafeIndexedDataset, safe_collate as drm_safe_collate, validate_checkpoint as validate_drm_only  # noqa: E402
from validate_image_sdf_checkpoints import validate_checkpoint as validate_image  # noqa: E402
from validate_mtm_z_checkpoints import validate_checkpoint as validate_mtm_z  # noqa: E402
from data import create_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drm_only_checkpoints", nargs="*", default=[])
    parser.add_argument("--drm_only_names", nargs="*", default=[])
    parser.add_argument("--image_checkpoints", nargs="*", default=[])
    parser.add_argument("--image_names", nargs="*", default=[])
    parser.add_argument("--mtm_z_checkpoints", nargs="*", default=[])
    parser.add_argument("--mtm_z_names", nargs="*", default=[])
    parser.add_argument("--dataroot", type=str, default="mpv3d_example")
    parser.add_argument("--datalist", type=str, default="val_pairs")
    parser.add_argument("--datamode", type=str, default="aligned")
    parser.add_argument("--dataset_model", type=str, default="MTM", choices=["MTM", "DRM"])
    parser.add_argument("--warproot", type=str, default="")
    parser.add_argument("--image_key", type=str, default="person")
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--num_surface_points", type=int, default=100000)
    parser.add_argument("--grid_shape", type=int, nargs=3, default=None)
    parser.add_argument("--allow_index_fallback", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output_csv", type=str, default="")
    add_mesh_eval_args(parser)
    parser.add_argument("--mtm_fusion_mode", type=str, default="", choices=["", "add", "replace", "concat"])
    parser.add_argument("--mtm_z_scale", type=float, default=None)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="all_checkpoint_validation")
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def check_names(kind, checkpoints, names):
    if names and len(names) != len(checkpoints):
        raise ValueError(f"--{kind}_names must match --{kind}_checkpoints length")


def default_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    args = parse_args()
    check_names("drm_only", args.drm_only_checkpoints, args.drm_only_names)
    check_names("image", args.image_checkpoints, args.image_names)
    check_names("mtm_z", args.mtm_z_checkpoints, args.mtm_z_names)
    if not (args.drm_only_checkpoints or args.image_checkpoints or args.mtm_z_checkpoints):
        raise ValueError("Pass at least one checkpoint group")

    device = get_device(args.gpu_id)
    print(f"Using device: {device}")
    wandb_run = init_wandb(args, "all_checkpoint_validation")
    rows = []

    if args.drm_only_checkpoints:
        drm_loader = create_dataset(build_dataset_opt(args))
        drm_base = drm_loader.dataset
        drm_dataset = SafeIndexedDataset(drm_base)
        print(f"DRM-only dataset size: wrapped={len(drm_dataset)} base={len(drm_base)} datalist={args.datalist}")
        drm_dataloader = DataLoader(drm_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=drm_safe_collate)
        for i, checkpoint in enumerate(args.drm_only_checkpoints):
            name = args.drm_only_names[i] if args.drm_only_names else default_name(checkpoint)
            print(f"Validating DRM-only {name}: {checkpoint}")
            rows.append(validate_drm_only(checkpoint, name, drm_dataloader, args, device))

    if args.image_checkpoints:
        image_dataset, image_base = create_image_sdf_dataset(args)
        print(f"Image-DRM dataset size: wrapped={len(image_dataset)} base={len(image_base)} datalist={args.datalist}")
        image_dataloader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=safe_collate)
        for i, checkpoint in enumerate(args.image_checkpoints):
            name = args.image_names[i] if args.image_names else default_name(checkpoint)
            print(f"Validating image-DRM {name}: {checkpoint}")
            rows.append(validate_image(checkpoint, name, image_dataloader, args, device, wandb_run=None))

    if args.mtm_z_checkpoints:
        mtm_dataset, mtm_base = create_mtm_dataset(args, is_train=True, serial_batches=True)
        print(f"MTM-z dataset size: wrapped={len(mtm_dataset)} base={len(mtm_base)} datalist={args.datalist}")
        mtm_dataloader = DataLoader(mtm_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=safe_collate)
        for i, checkpoint in enumerate(args.mtm_z_checkpoints):
            name = args.mtm_z_names[i] if args.mtm_z_names else default_name(checkpoint)
            print(f"Validating MTM-z {name}: {checkpoint}")
            rows.append(validate_mtm_z(checkpoint, name, mtm_dataloader, args, device))

    print_and_save_results(rows, args, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
