#!/usr/bin/env python3
"""Validate MTM-z + image encoder + DRM checkpoints."""

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

from models_2 import DRMSDFModel  # noqa: E402
from tools_2_image_encoder.common import get_device, prepare_sdf_batch  # noqa: E402
from tools_2_image_encoder.image_encoder_model import build_image_encoder_from_args  # noqa: E402
from tools_2_mtm_z.common import build_mtm, create_mtm_dataset, prepare_mtm_inputs, safe_collate  # noqa: E402
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
    parser.add_argument("--warproot", type=str, default="")
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
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--mtm_fusion_mode", type=str, default="", choices=["", "add", "replace", "concat"])
    parser.add_argument("--mtm_z_scale", type=float, default=None)
    add_mesh_eval_args(parser)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="mtm_z_validation")
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def image_encoder_runtime(config):
    image_config = config.get("image_drm_config", {})
    image_latent_dim = int(config.get("image_latent_dim", image_config.get("latent_dim", config.get("latent_dim", 128))))
    return SimpleNamespace(
        latent_dim=image_latent_dim,
        image_channels=int(image_config.get("image_channels", 3)),
        encoder_base_channels=int(image_config.get("encoder_base_channels", 32)),
        encoder_num_blocks=int(image_config.get("encoder_num_blocks", 5)),
        encoder_head_hidden_dim=int(image_config.get("encoder_head_hidden_dim", 512)),
        encoder_dropout=float(image_config.get("encoder_dropout", 0.0)),
        encoder_use_batchnorm=int(image_config.get("encoder_use_batchnorm", 1)),
    )


def set_mtm_args_from_config(args, config):
    for key, default in (
        ("mtm_input_nc_A", 29),
        ("mtm_input_nc_B", 3),
        ("mtm_ngf", 64),
        ("mtm_n_layers_feat_extract", 3),
        ("mtm_grid_size", 3),
        ("mtm_add_tps", 0),
        ("mtm_add_depth", 0),
        ("mtm_add_segmt", 0),
        ("mtm_norm", "instance"),
        ("mtm_use_dropout", 0),
        ("mtm_init_type", "normal"),
        ("mtm_init_gain", 0.02),
    ):
        setattr(args, key, config.get(key, default))


def fuse_latents(z_image, z_mtm, config):
    mode = config.get("mtm_fusion_mode", "add")
    scale = float(config.get("mtm_z_scale", 1.0))
    scaled_mtm = scale * z_mtm
    if mode == "replace":
        return scaled_mtm
    if mode == "concat":
        return torch.cat([z_image, scaled_mtm], dim=-1)
    return z_image + scaled_mtm


def load_mtm_z_checkpoint(path, args, device):
    checkpoint = torch.load(path, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    if args.mtm_fusion_mode:
        config["mtm_fusion_mode"] = args.mtm_fusion_mode
    if args.mtm_z_scale is not None:
        config["mtm_z_scale"] = args.mtm_z_scale

    image_config = config.get("image_drm_config", {})
    image_latent_dim = int(config.get("image_latent_dim", image_config.get("latent_dim", config.get("latent_dim", 128))))
    mtm_latent_dim = int(config.get("mtm_latent_dim", image_latent_dim))
    drm_latent_dim = int(config.get("drm_latent_dim", config.get("latent_dim", image_latent_dim)))
    set_mtm_args_from_config(args, config)

    mtm = build_mtm(args, latent_dim=mtm_latent_dim, device=device)
    mtm.load_state_dict(checkpoint["mtm_state"], strict=False)
    encoder = build_image_encoder_from_args(image_encoder_runtime(config)).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    drm = DRMSDFModel(
        latent_dim=drm_latent_dim,
        point_dim=3,
        hidden_dim=int(config["sdf_hidden_dim"]),
        num_layers=int(config["sdf_num_layers"]),
        pe_L=int(config["pe_L"]),
    ).to(device)
    drm.load_state_dict(checkpoint["drm_state"])
    mtm.eval()
    encoder.eval()
    drm.eval()
    print(
        f"Loaded MTM-z checkpoint with fusion={config.get('mtm_fusion_mode', 'add')} "
        f"scale={config.get('mtm_z_scale', 1.0)} image_latent={image_latent_dim} "
        f"mtm_latent={mtm_latent_dim} drm_latent={drm_latent_dim}"
    )
    return mtm, encoder, drm, config


def validate_checkpoint(path, model_name, dataloader, args, device):
    mtm, encoder, drm, config = load_mtm_z_checkpoint(path, args, device)
    meter = RunningSDFMetrics(model_name, args)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            meter.seen_batches += 1
            prepared = prepare_sdf_batch(batch, device)
            mtm_inputs = prepare_mtm_inputs(batch, device)
            if prepared is None or mtm_inputs is None:
                meter.add_empty_batch()
                continue
            agnostic, cloth = mtm_inputs
            z_mtm = mtm(agnostic, cloth).get("z", None)
            if z_mtm is None:
                raise RuntimeError("MTM forward did not return output['z']")
            z_image = encoder(prepared["images"]).unsqueeze(1)
            z = fuse_latents(z_image, z_mtm, config)
            pred_sdf = drm(z, prepared["points"])
            meter.add_batch(pred_sdf, prepared["sdf_gt"], points=prepared["points"])
            names = list(batch.get("im_name", []))
            for item_index in range(z.shape[0]):
                sample_name = names[item_index] if item_index < len(names) else f"sample_{batch_index}_{item_index}.png"
                maybe_add_mesh_metric(
                    meter,
                    drm,
                    z[item_index:item_index + 1],
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
        raise ValueError("--model_names must have same length as --checkpoints")
    device = get_device(args.gpu_id)
    print(f"Using device: {device}")
    dataset, base_dataset = create_mtm_dataset(args, is_train=True, serial_batches=True)
    print(f"Validation dataset size: wrapped={len(dataset)} base={len(base_dataset)} datalist={args.datalist}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=safe_collate)
    wandb_run = init_wandb(args, "mtm_z_validation")
    rows = []
    for i, checkpoint in enumerate(args.checkpoints):
        name = args.model_names[i] if args.model_names else os.path.splitext(os.path.basename(checkpoint))[0]
        print(f"Validating {name}: {checkpoint}")
        rows.append(validate_checkpoint(checkpoint, name, dataloader, args, device))
    print_and_save_results(rows, args, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
