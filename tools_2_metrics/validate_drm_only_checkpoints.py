#!/usr/bin/env python3
"""Validate DRM-only latent-codebook checkpoints.

DRM-only checkpoints can only evaluate samples that exist in the checkpoint's
saved sample_names/latent codebook. Unseen images have no latent vector and are
skipped.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for path in (PROJECT_ROOT, CURRENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import create_dataset  # noqa: E402
from models_2 import DRMSDFModel, LatentCodebook  # noqa: E402
from tools_2_image_encoder.common import ensure_shape_sdf, get_device  # noqa: E402
from validation_common import (  # noqa: E402
    RunningSDFMetrics,
    add_mesh_eval_args,
    init_wandb,
    maybe_add_mesh_metric,
    print_and_save_results,
    tensor_item_points,
)


class SafeIndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        try:
            sample = self.base_dataset[idx]
        except FileNotFoundError as exc:
            print(f"[skip sample] index={idx}: missing file: {exc}")
            return None
        except OSError as exc:
            print(f"[skip sample] index={idx}: failed to load file: {exc}")
            return None
        sample["dataset_index"] = idx
        return sample


def safe_collate(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    return default_collate(valid)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", "--checkpoint", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="*", default=None)
    parser.add_argument("--dataroot", type=str, default="mpv3d_example")
    parser.add_argument("--datalist", type=str, default="val_pairs")
    parser.add_argument("--datamode", type=str, default="aligned")
    parser.add_argument("--dataset_model", type=str, default="MTM", choices=["MTM", "DRM"])
    parser.add_argument("--warproot", type=str, default="")
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--num_surface_points", type=int, default=100000)
    parser.add_argument("--grid_shape", type=int, nargs=3, default=None)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--allow_index_fallback", type=int, default=0, choices=[0, 1])
    add_mesh_eval_args(parser)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="drm_only_validation")
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def build_dataset_opt(args):
    return SimpleNamespace(
        dataroot=args.dataroot,
        datalist=args.datalist,
        datamode=args.datamode,
        model=args.dataset_model,
        batch_size=args.batch_size,
        img_width=args.img_width,
        img_height=args.img_height,
        isTrain=True,
        max_dataset_size=float("inf"),
        num_threads=args.num_workers,
        serial_batches=True,
        no_pin_memory=False,
        radius=args.radius,
        warproot=args.warproot,
    )


def load_drm_only_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    latent_state = checkpoint.get("latent_state")
    model_state = checkpoint.get("model_state") or checkpoint.get("drm_state")
    if latent_state is None or model_state is None:
        raise RuntimeError(f"{path} is not a DRM-only checkpoint with model_state and latent_state")
    latent_weight = latent_state.get("embedding.weight")
    if latent_weight is None:
        raise RuntimeError(f"{path} latent_state is missing embedding.weight")
    latent_dim = int(config.get("latent_dim", latent_weight.shape[1]))
    num_embeddings = int(latent_weight.shape[0])
    model = DRMSDFModel(
        latent_dim=latent_dim,
        point_dim=3,
        hidden_dim=int(config.get("sdf_hidden_dim", 512)),
        num_layers=int(config.get("sdf_num_layers", 8)),
        pe_L=int(config.get("pe_L", 6)),
    ).to(device)
    latent_codebook = LatentCodebook(num_embeddings=num_embeddings, latent_dim=latent_dim).to(device)
    model.load_state_dict(model_state)
    latent_codebook.load_state_dict(latent_state)
    model.eval()
    latent_codebook.eval()
    sample_names = list(checkpoint.get("sample_names", []))
    name_to_index = {name: i for i, name in enumerate(sample_names)}
    return model, latent_codebook, sample_names, name_to_index


def batch_sample_names(batch):
    names = batch.get("im_name", None)
    if names is None:
        return []
    return list(names)


def validate_checkpoint(path, model_name, dataloader, args, device):
    model, latent_codebook, sample_names, name_to_index = load_drm_only_checkpoint(path, device)
    meter = RunningSDFMetrics(model_name, args)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            meter.seen_batches += 1
            if batch is None:
                meter.add_empty_batch()
                continue
            points, sdf_gt = ensure_shape_sdf(batch.get("sdf_points"), batch.get("sdf_gt"))
            if points is None or sdf_gt is None or points.size(1) == 0:
                meter.add_empty_batch()
                continue
            names = batch_sample_names(batch)
            fallback_indices = batch.get("dataset_index")
            latent_indices = []
            keep = []
            for i, name in enumerate(names):
                if name in name_to_index:
                    latent_indices.append(name_to_index[name])
                    keep.append(i)
                elif args.allow_index_fallback and fallback_indices is not None and int(fallback_indices[i]) < len(sample_names):
                    latent_indices.append(int(fallback_indices[i]))
                    keep.append(i)
            skipped = points.shape[0] - len(keep)
            if skipped > 0:
                meter.add_skipped_samples(skipped)
            if not keep:
                meter.add_empty_batch()
                continue
            keep_t = torch.as_tensor(keep, dtype=torch.long)
            raw_surface = batch.get("surface_points", None)
            points = points[keep_t].to(device)
            sdf_gt = sdf_gt[keep_t].to(device)
            surface_points = raw_surface[keep_t] if isinstance(raw_surface, torch.Tensor) and raw_surface.dim() >= 3 else None
            latent_indices = torch.as_tensor(latent_indices, dtype=torch.long, device=device)
            z = latent_codebook(latent_indices).unsqueeze(1)
            pred_sdf = model(z, points)
            meter.add_batch(pred_sdf, sdf_gt, points=points)
            kept_names = [names[i] for i in keep]
            for item_index, sample_name in enumerate(kept_names):
                maybe_add_mesh_metric(
                    meter,
                    model,
                    z[item_index:item_index + 1],
                    args,
                    device,
                    sample_name,
                    surface_points=tensor_item_points(surface_points, item_index),
                    sdf_points=tensor_item_points(points, item_index),
                )
    if meter.skipped_samples > 0:
        print(f"[{model_name}] skipped samples with no saved latent code. DRM-only cannot evaluate unseen images.")
    return meter.finalize()


def main():
    args = parse_args()
    if args.model_names and len(args.model_names) != len(args.checkpoints):
        raise ValueError("--model_names must have same length as --checkpoints")
    device = get_device(args.gpu_id)
    print(f"Using device: {device}")
    dataset_loader = create_dataset(build_dataset_opt(args))
    base_dataset = dataset_loader.dataset
    dataset = SafeIndexedDataset(base_dataset)
    print(f"Validation dataset size: wrapped={len(dataset)} base={len(base_dataset)} datalist={args.datalist}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=safe_collate)
    wandb_run = init_wandb(args, "drm_only_validation")
    rows = []
    for i, checkpoint in enumerate(args.checkpoints):
        name = args.model_names[i] if args.model_names else os.path.splitext(os.path.basename(checkpoint))[0]
        print(f"Validating {name}: {checkpoint}")
        rows.append(validate_checkpoint(checkpoint, name, dataloader, args, device))
    print_and_save_results(rows, args, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
