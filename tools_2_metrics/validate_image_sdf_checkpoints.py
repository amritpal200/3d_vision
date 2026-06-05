#!/usr/bin/env python3
"""Validate and compare image-conditioned SDF checkpoints."""

import argparse
import csv
import math
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

from sdf_metrics import evaluate_sdf, format_metrics_table, is_better_metrics, log_metrics_to_wandb
from common import create_image_sdf_dataset, get_device, prepare_sdf_batch, safe_collate
from image_encoder_model import build_image_encoder_from_args
from models_2 import DRMSDFModel


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


def write_csv(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "model",
        "near_surface_mae",
        "sign_accuracy",
        "chamfer_distance",
        "f_score",
        "normal_consistency",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def nanmean(values):
    finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def validate_checkpoint(checkpoint_path, model_name, dataloader, args, device, wandb_run=None):
    encoder, drm = load_image_sdf_checkpoint(checkpoint_path, device)

    near_abs_sum = 0.0
    near_count = 0
    sign_correct = 0.0
    total_count = 0
    mesh_metric_values = {
        "chamfer_distance": [],
        "f_score": [],
        "normal_consistency": [],
    }
    mesh_warning_printed = False
    seen_batches = 0
    valid_batches = 0
    empty_batches = 0
    gt_min = float("inf")
    gt_max = float("-inf")

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            seen_batches += 1

            prepared = prepare_sdf_batch(batch, device)
            if prepared is None:
                empty_batches += 1
                continue
            valid_batches += 1

            latent_z = encoder(prepared["images"]).unsqueeze(1)
            pred_sdf = drm(latent_z, prepared["points"])
            gt_sdf = prepared["sdf_gt"]

            pred_flat = pred_sdf.reshape(-1)
            gt_flat = gt_sdf.reshape(-1)
            if gt_flat.numel() > 0:
                gt_min = min(gt_min, float(gt_flat.min().detach().item()))
                gt_max = max(gt_max, float(gt_flat.max().detach().item()))
            near_mask = torch.abs(gt_flat) < float(args.tau)
            if near_mask.any():
                near_abs_sum += torch.abs(pred_flat[near_mask] - gt_flat[near_mask]).sum().item()
                near_count += int(near_mask.sum().item())

            sign_correct += ((pred_flat < 0) == (gt_flat < 0)).float().sum().item()
            total_count += int(gt_flat.numel())

            if args.grid_shape is not None:
                batch_size = pred_sdf.shape[0]
                for item_index in range(batch_size):
                    metrics, details = evaluate_sdf(
                        pred_sdf[item_index].detach().cpu(),
                        gt_sdf[item_index].detach().cpu(),
                        points=prepared["points"][item_index].detach().cpu(),
                        grid_shape=args.grid_shape,
                        tau=args.tau,
                        delta=args.delta,
                        num_surface_points=args.num_surface_points,
                        return_details=True,
                    )
                    for key in mesh_metric_values:
                        mesh_metric_values[key].append(metrics[key])
                    if not mesh_warning_printed and "mesh_metrics_error" in details:
                        print(f"[{model_name}] mesh metrics warning: {details['mesh_metrics_error']}")
                        mesh_warning_printed = True

    metrics = {
        "near_surface_mae": near_abs_sum / near_count if near_count > 0 else float("nan"),
        "sign_accuracy": sign_correct / total_count if total_count > 0 else float("nan"),
        "chamfer_distance": nanmean(mesh_metric_values["chamfer_distance"]),
        "f_score": nanmean(mesh_metric_values["f_score"]),
        "normal_consistency": nanmean(mesh_metric_values["normal_consistency"]),
    }

    gt_range = "empty" if total_count == 0 else f"[{gt_min:.6g}, {gt_max:.6g}]"
    print(
        f"[{model_name}] validation diagnostics: "
        f"seen_batches={seen_batches}, valid_batches={valid_batches}, empty_batches={empty_batches}, "
        f"total_sdf_points={total_count}, near_surface_points={near_count}, gt_sdf_range={gt_range}"
    )
    if total_count == 0:
        print(
            f"[{model_name}] no SDF validation points were loaded. Check --datalist, dataset split, "
            f"and whether samples contain sdf_points/sdf_gt."
        )
    if near_count == 0 and total_count > 0:
        print(f"[{model_name}] no near-surface validation points found with tau={args.tau}; try a larger --tau such as 0.05")
    if args.grid_shape is None:
        print(f"[{model_name}] mesh metrics skipped: pass --grid_shape nz ny nx for dense-grid SDF validation")

    log_metrics_to_wandb(metrics, prefix=f"val/{model_name}", wandb_run=wandb_run)
    return {"model": model_name, **metrics}


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

    wandb_run = None
    if args.wandb_mode != "disabled" and args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=vars(args),
            )
        except Exception as exc:
            print(f"wandb init failed; continuing without wandb: {exc}")

    rows = []
    best_row = None
    for index, checkpoint_path in enumerate(args.checkpoints):
        model_name = args.model_names[index] if args.model_names else os.path.splitext(os.path.basename(checkpoint_path))[0]
        print(f"Validating {model_name}: {checkpoint_path}")
        row = validate_checkpoint(checkpoint_path, model_name, dataloader, args, device, wandb_run=wandb_run)
        rows.append(row)
        if is_better_metrics(row, best_row):
            best_row = row

    print(format_metrics_table(rows))
    if best_row is not None:
        print(f"Best model by priority: {best_row['model']}")

    write_csv(args.output_csv, rows)
    if args.output_csv:
        print(f"Wrote metrics CSV: {args.output_csv}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

