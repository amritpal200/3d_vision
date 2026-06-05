
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 parcial/train_drm_only.py --data_dir /data/113-1/users/asingh/humanMesh_dataset/output --checkpoints_dir /data/113-1/users/asingh/project/3d/checkpoints/parcial --name parcial_overfit --batch_size 10 --num_epochs 500 --points_per_sample 65536 --latent_dim 256 --sdf_hidden_dim 512 --sdf_num_layers 8 --pe_L 6 --wandb_project parcial_sdf --wandb_name parcial_overfit_run --wandb_mode online


#!/usr/bin/env python3
"""Train a DRM-only SDF model on locally precomputed mesh samples.

This trainer is meant for a tiny custom dataset, such as a few human meshes
converted to `.npz` files containing `points` and `sdf` arrays.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

from drm_only_model import DRMSDFModel, LatentCodebook, build_checkpoint
from sdf_dataset import SDFDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing .npz SDF samples")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--name", type=str, default="parcial_overfit")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--lr_model", type=float, default=1e-4)
    parser.add_argument("--lr_latent", type=float, default=3e-4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--sdf_hidden_dim", type=int, default=512)
    parser.add_argument("--sdf_num_layers", type=int, default=8)
    parser.add_argument("--pe_L", type=int, default=6)
    parser.add_argument("--points_per_sample", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="parcial_sdf")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=1)
    return parser.parse_args()


def ensure_batch_shapes(points, sdf_gt):
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if sdf_gt.dim() == 2:
        sdf_gt = sdf_gt.unsqueeze(0)
    if sdf_gt.dim() == 1:
        sdf_gt = sdf_gt.view(1, -1, 1)
    return points, sdf_gt


def init_wandb(args):
    if wandb is None or args.wandb_mode == "disabled":
        print("wandb disabled or unavailable; continuing without logging")
        return None

    run_name = args.wandb_name.strip() or args.name
    try:
        return wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            mode=args.wandb_mode,
        )
    except Exception:
        print("wandb init failed; continuing without logging")
        return None


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device(args.gpu_id)
    print(f"Using device: {device}")

    dataset = SDFDataset(
        args.data_dir,
        points_per_sample=args.points_per_sample,
        seed=args.seed,
    )

    # select only the first sample for overfitting
    dataset.files = dataset.files[:1]

    if len(dataset) == 0:
        raise RuntimeError(f"No .npz files found in {args.data_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = DRMSDFModel(
        latent_dim=args.latent_dim,
        point_dim=3,
        hidden_dim=args.sdf_hidden_dim,
        num_layers=args.sdf_num_layers,
        pe_L=args.pe_L,
    ).to(device)
    latent_codebook = LatentCodebook(
        num_embeddings=len(dataset),
        latent_dim=args.latent_dim,
        init_std=0.02,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr_model},
            {"params": latent_codebook.parameters(), "lr": args.lr_latent},
        ],
        betas=(0.5, 0.999),
    )

    save_dir = os.path.join(args.checkpoints_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)

    wandb_run = init_wandb(args)

    best_loss = float("inf")
    global_step = 0
    max_steps = args.max_steps if args.max_steps > 0 else float("inf")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        latent_codebook.train()
        running_loss = 0.0
        valid_batches = 0

        for batch in dataloader:
            if global_step >= max_steps:
                break

            points = batch["sdf_points"]
            sdf_gt = batch["sdf_gt"]
            sample_indices = batch["sample_index"].long()

            points, sdf_gt = ensure_batch_shapes(points, sdf_gt)
            points = points.to(device)
            sdf_gt = sdf_gt.to(device)
            sample_indices = sample_indices.to(device)

            z = latent_codebook(sample_indices).unsqueeze(1)
            # sdf_pred = model(z, points)
            # loss = F.mse_loss(sdf_pred, sdf_gt)
            sdf_pred = model(z, points)
            loss_sdf = F.mse_loss(sdf_pred, sdf_gt)
            loss_latent = 1e-4 * (z ** 2).mean()
            loss = loss_sdf + loss_latent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            valid_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                print(
                    f"epoch={epoch}/{args.num_epochs} step={global_step} "
                    f"loss={loss.detach().item():.6f} "
                    f"loss_sdf={loss_sdf.detach().item():.6f} "
                    f"loss_latent={loss_latent.detach().item():.6f}"
                )

            if wandb_run is not None and global_step % max(1, args.wandb_log_every) == 0:
                try:
                    wandb.log(
                        {
                            "train/loss": loss.detach().item(),
                            "train/loss_sdf": loss_sdf.detach().item(),
                            "train/loss_latent": loss_latent.detach().item(),
                            "train/epoch": epoch,
                            "train/step": global_step,
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

        if valid_batches == 0:
            print(f"epoch={epoch}: no valid batches found")
            continue

        epoch_loss = running_loss / float(valid_batches)
        print(f"epoch={epoch} mean_loss={epoch_loss:.6f} valid_batches={valid_batches}")

        config = vars(args)
        checkpoint = build_checkpoint(
            model=model,
            latent_codebook=latent_codebook,
            epoch=epoch,
            global_step=global_step,
            config=config,
            sample_names=dataset.sample_names,
        )

        latest_path = os.path.join(save_dir, "latest_net_DRMOnly.pth")
        torch.save(checkpoint, latest_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(save_dir, "best_net_DRMOnly.pth")
            torch.save(checkpoint, best_path)
            print(f"Updated best checkpoint: {best_path} (mean_loss={best_loss:.6f})")

            if wandb_run is not None:
                try:
                    wandb.log(
                        {
                            "epoch/best_loss": best_loss,
                            "epoch": epoch,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

        if epoch % args.save_every == 0:
            epoch_path = os.path.join(save_dir, f"epoch_{epoch}_net_DRMOnly.pth")
            torch.save(checkpoint, epoch_path)

        if global_step >= max_steps:
            print(f"Reached --max_steps={args.max_steps}, stopping early.")
            break

    print("Training finished.")
    print(f"Checkpoints saved under: {save_dir}")

    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()