#!/usr/bin/env python3
"""Validate and compare image-conditioned SDF checkpoints."""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
IMAGE_ENCODER_DIR = os.path.join(PROJECT_ROOT, "tools_2_image_encoder")
for path in (PROJECT_ROOT, CURRENT_DIR, IMAGE_ENCODER_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from tools_2_image_encoder.common import create_image_sdf_dataset, get_device, prepare_sdf_batch, safe_collate  # noqa: E402
from tools_2_image_encoder.image_encoder_model import build_image_encoder_from_args  # noqa: E402
from models_2 import DRMSDFModel  # noqa: E402
from validation_common import (  # noqa: E402
    RunningSDFMetrics,
    add_mesh_eval_args,
    init_wandb,
    maybe_add_mesh_metric,
    print_and_save_results,
    tensor_item_points,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", "--checkpoint", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="*", default=None)

    parser.add_argument("--dataroot", type=str, default="mpv3d_example")
    parser.add_argument("--datalist", type=str, default="val_pairs")
    parser.add_argument("--datamode", type=str, default="aligned")
    parser.add_argument("--dataset_model", type=str, default="MTM", choices=["MTM", "DRM"])
    parser.add_argument("--warproot", type=str, default="")
    parser.add_argument("--image_key", type=str, default="person")
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--num_surface_points", type=int, default=100000)
    parser.add_argument("--grid_shape", type=int, nargs=3, default=None, help="Optional dense grid shape: nz ny nx")
    parser.add_argument("--output_csv", type=str, default="")
    add_mesh_eval_args(parser)

    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="image_sdf_checkpoint_validation")
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def runtime_from_config(config):
    return SimpleNamespace(
        latent_dim=int(config.get("latent_dim", 128)),
        sdf_hidden_dim=int(config.get("sdf_hidden_dim", 512)),
        sdf_num_layers=int(config.get("sdf_num_layers", 8)),
        pe_L=int(config.get("pe_L", 6)),
        image_channels=int(config.get("image_channels", 3)),
        encoder_base_channels=int(config.get("encoder_base_channels", 32)),
        encoder_num_blocks=int(config.get("encoder_num_blocks", 5)),
        encoder_head_hidden_dim=int(config.get("encoder_head_hidden_dim", 512)),
        encoder_dropout=float(config.get("encoder_dropout", 0.0)),
        encoder_use_batchnorm=int(config.get("encoder_use_batchnorm", 1)),
    )


def load_image_sdf_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    runtime = runtime_from_config(config)

    encoder = build_image_encoder_from_args(runtime).to(device)
    drm = DRMSDFModel(
        latent_dim=runtime.latent_dim,
        point_dim=3,
        hidden_dim=runtime.sdf_hidden_dim,
        num_layers=runtime.sdf_num_layers,
        pe_L=runtime.pe_L,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state"])
    drm_state = checkpoint.get("drm_state", checkpoint.get("model_state"))
    if drm_state is None:
        raise RuntimeError(f"{checkpoint_path} is missing drm_state/model_state")
    drm.load_state_dict(drm_state)
    encoder.eval()
    drm.eval()
    return encoder, drm


def validate_checkpoint(checkpoint_path, model_name, dataloader, args, device, wandb_run=None):
    encoder, drm = load_image_sdf_checkpoint(checkpoint_path, device)
    meter = RunningSDFMetrics(model_name, args)

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            meter.seen_batches += 1

            prepared = prepare_sdf_batch(batch, device)
            if prepared is None:
                meter.add_empty_batch()
                continue

            latent_z = encoder(prepared["images"]).unsqueeze(1)
            pred_sdf = drm(latent_z, prepared["points"])
            meter.add_batch(pred_sdf, prepared["sdf_gt"], points=prepared["points"])

            names = list(batch.get("im_name", []))
            for item_index in range(latent_z.shape[0]):
                sample_name = names[item_index] if item_index < len(names) else f"sample_{batch_index}_{item_index}.png"
                maybe_add_mesh_metric(
                    meter,
                    drm,
                    latent_z[item_index:item_index + 1],
                    args,
                    device,
                    sample_name,
                    surface_points=tensor_item_points(prepared.get("surface_points"), item_index),
                    sdf_points=tensor_item_points(prepared.get("points"), item_index),
                )

    return meter.finalize()


def main():
    args = parse_args()
    if args.model_names and len(args.model_names) != len(args.checkpoints):
        raise ValueError("--model_names must have the same length as --checkpoints")

    device = get_device(args.gpu_id)
    print(f"Using device: {device}")

    dataset, base_dataset = create_image_sdf_dataset(args)
    print(f"Validation dataset size: wrapped={len(dataset)} base={len(base_dataset)} datalist={args.datalist}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate,
    )

    wandb_run = init_wandb(args, "image_sdf_checkpoint_validation")
    rows = []
    for index, checkpoint_path in enumerate(args.checkpoints):
        model_name = args.model_names[index] if args.model_names else os.path.splitext(os.path.basename(checkpoint_path))[0]
        print(f"Validating {model_name}: {checkpoint_path}")
        rows.append(validate_checkpoint(checkpoint_path, model_name, dataloader, args, device, wandb_run=wandb_run))

    print_and_save_results(rows, args, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
