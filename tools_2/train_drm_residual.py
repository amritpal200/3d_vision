
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 tools_2/train_drm_residual.py --dataroot /data/113-1/users/asingh/project/3d/MPV3D --datalist train_pairs --pretrained_coarse_checkpoint /data/113-1/users/asingh/project/3d/checkpoints/3rd_version/aligned/DRM_only_bootstrap/best_net_DRMOnly.pth --name DRM_residual_final_full --freeze_coarse 1 --lr_coarse 1e-5 --lr_residual 1e-4 --supervision_mode final  --batch_size 6 --num_epochs 200 --checkpoints_dir /data/113-1/users/asingh/project/3d/checkpoints/residual_corase_train_residual/ --latent_dim 128 --sdf_hidden_dim 512 --pe_L 6


#!/usr/bin/env python3
"""
Train a coarse DRM plus a residual DRM refinement branch.

Modes:
1) Freeze coarse model, train only residual + latent:
   --freeze_coarse 1

2) Fine-tune coarse + residual + latent:
   --freeze_coarse 0

Supervision modes:
1) Explicit residual learning:
   residual_pred ≈ sdf_gt - coarse_pred
   --supervision_mode residual

2) Direct final SDF learning:
   coarse_pred + residual_pred ≈ sdf_gt
   --supervision_mode final
"""

import argparse
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except Exception:
    wandb = None

sys.path.append(".")

from data import create_dataset
from models_2 import DRMSDFModel, LatentCodebook


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample["sample_index"] = idx
        return sample


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_opt(args):
    opt = SimpleNamespace()
    opt.dataroot = args.dataroot
    opt.datalist = args.datalist
    opt.datamode = args.datamode
    opt.model = args.dataset_model
    opt.batch_size = args.batch_size
    opt.img_width = args.img_width
    opt.img_height = args.img_height
    opt.isTrain = True
    opt.max_dataset_size = float("inf")
    opt.num_threads = args.num_workers
    opt.serial_batches = False
    opt.no_pin_memory = False
    opt.radius = args.radius
    opt.warproot = args.warproot
    return opt


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def ensure_shape_sdf(points, sdf_gt):
    if points is None or sdf_gt is None:
        return None, None

    if points.dim() == 2:
        points = points.unsqueeze(0)

    if sdf_gt.dim() == 1:
        sdf_gt = sdf_gt.view(1, -1, 1)
    elif sdf_gt.dim() == 2:
        sdf_gt = sdf_gt.unsqueeze(-1)

    return points, sdf_gt


def predict_with_grad(model_fn, latent_z, points):
    points_req = points.clone().detach().requires_grad_(True)
    sdf_pred = model_fn(latent_z, points_req)

    grads = torch.autograd.grad(
        outputs=sdf_pred.sum(),
        inputs=points_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return sdf_pred, grads


def make_zero_like(reference):
    return torch.zeros(1, device=reference.device, dtype=reference.dtype)


def compute_regression_loss(pred, target, loss_type):
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "l2":
        return F.mse_loss(pred, target)
    if loss_type == "smoothl1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unknown loss type: {loss_type}")


def init_wandb(args):
    if wandb is None or args.wandb_mode == "disabled":
        print("wandb not available or disabled; continuing without remote logging")
        return None

    run_name = args.wandb_name.strip() or args.name

    try:
        return wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=dict(vars(args)),
            mode=args.wandb_mode,
        )
    except Exception as exc:
        print(f"wandb init failed; continuing without remote logging: {exc}")
        return None


def load_pretrained_coarse(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        elif "coarse_state" in checkpoint:
            model_state = checkpoint["coarse_state"]
        else:
            model_state = checkpoint
    else:
        model_state = checkpoint

    if hasattr(model_state, "_metadata"):
        del model_state._metadata

    load_result = model.load_state_dict(model_state, strict=True)
    model.to(device)

    print(f"Loaded pretrained coarse checkpoint: {checkpoint_path}")
    print(load_result)

    return checkpoint, load_result


def set_trainable(model, trainable):
    for p in model.parameters():
        p.requires_grad = trainable

    if trainable:
        model.train()
    else:
        model.eval()


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataroot", type=str, default="mpv3d_example")
    parser.add_argument("--datalist", type=str, default="train_pairs")
    parser.add_argument("--datamode", type=str, default="aligned")
    parser.add_argument("--dataset_model", type=str, default="MTM", choices=["MTM", "DRM"])
    parser.add_argument("--warproot", type=str, default="")
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=5)

    # Output
    parser.add_argument("--name", type=str, default="DRM_residual_bootstrap")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1)

    # Checkpoint
    parser.add_argument("--pretrained_coarse_checkpoint", type=str, required=True)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)

    # Freeze / train mode
    parser.add_argument(
        "--freeze_coarse",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 = freeze pretrained coarse DRM, 0 = fine-tune coarse DRM",
    )

    parser.add_argument(
        "--train_latent",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 = train latent codebook, 0 = freeze latent codebook",
    )

    # Optimizer
    parser.add_argument("--lr_coarse", type=float, default=1e-5)
    parser.add_argument("--lr_residual", type=float, default=1e-4)
    parser.add_argument("--lr_latent", type=float, default=3e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.5)
    parser.add_argument("--adam_beta2", type=float, default=0.999)

    # Model architecture
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--sdf_hidden_dim", type=int, default=512)
    parser.add_argument("--sdf_num_layers", type=int, default=8)
    parser.add_argument("--pe_L", type=int, default=6)

    # Main supervision
    parser.add_argument(
        "--supervision_mode",
        type=str,
        default="final",
        choices=["residual", "final", "both"],
        help="residual = supervise residual_pred, final = supervise final_pred, both = use both",
    )

    parser.add_argument(
        "--detach_coarse_for_residual",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 = residual target uses coarse_pred.detach()",
    )

    parser.add_argument(
        "--main_loss",
        type=str,
        default="l1",
        choices=["l1", "l2", "smoothl1"],
        help="Loss type for final/residual supervision",
    )

    parser.add_argument(
        "--coarse_loss",
        type=str,
        default="l1",
        choices=["l1", "l2", "smoothl1"],
    )

    # Loss weights
    parser.add_argument("--lambda_final", type=float, default=1.0)
    parser.add_argument("--lambda_residual", type=float, default=1.0)
    parser.add_argument("--lambda_coarse", type=float, default=0.0)
    parser.add_argument("--lambda_surface", type=float, default=0.1)
    parser.add_argument("--lambda_sign", type=float, default=0.1)
    parser.add_argument("--lambda_eikonal", type=float, default=0.1)
    parser.add_argument("--lambda_normal", type=float, default=0.1)

    # Loss switches
    parser.add_argument("--use_surface_loss", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_sign_loss", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_eikonal_loss", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_normal_loss", type=int, default=1, choices=[0, 1])

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="m3d_drm")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb_log_every", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device(args.gpu_id)
    print(f"Using device: {device}")

    ds_opt = build_dataset_opt(args)
    dataset_loader = create_dataset(ds_opt)
    base_dataset = dataset_loader.dataset
    dataset = IndexedDataset(base_dataset)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check --dataroot and --datalist.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    sample_names = getattr(base_dataset, "im_names", [str(i) for i in range(len(dataset))])

    coarse_model = DRMSDFModel(
        latent_dim=args.latent_dim,
        point_dim=3,
        hidden_dim=args.sdf_hidden_dim,
        num_layers=args.sdf_num_layers,
        pe_L=args.pe_L,
    ).to(device)

    checkpoint, _ = load_pretrained_coarse(
        coarse_model,
        args.pretrained_coarse_checkpoint,
        device,
    )

    residual_model = DRMSDFModel(
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

    if isinstance(checkpoint, dict) and "latent_state" in checkpoint:
        try:
            latent_codebook.load_state_dict(checkpoint["latent_state"])
            print("Loaded latent codebook from coarse checkpoint.")
        except Exception as exc:
            print(f"Could not load latent_state from checkpoint, keeping random init: {exc}")

    # Freeze or train coarse model
    if args.freeze_coarse == 1:
        set_trainable(coarse_model, False)
        print("Coarse DRM is frozen.")
    else:
        set_trainable(coarse_model, True)
        print("Coarse DRM is trainable.")

    residual_model.train()

    if args.train_latent == 1:
        set_trainable(latent_codebook, True)
        print("Latent codebook is trainable.")
    else:
        set_trainable(latent_codebook, False)
        print("Latent codebook is frozen.")

    param_groups = []

    if args.freeze_coarse == 0:
        param_groups.append(
            {
                "params": [p for p in coarse_model.parameters() if p.requires_grad],
                "lr": args.lr_coarse,
            }
        )

    param_groups.append(
        {
            "params": residual_model.parameters(),
            "lr": args.lr_residual,
        }
    )

    if args.train_latent == 1:
        param_groups.append(
            {
                "params": [p for p in latent_codebook.parameters() if p.requires_grad],
                "lr": args.lr_latent,
            }
        )

    optimizer = torch.optim.Adam(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    save_dir = os.path.join(args.checkpoints_dir, args.datamode, args.name)
    os.makedirs(save_dir, exist_ok=True)

    wandb_run = init_wandb(args)

    best_loss = float("inf")
    global_step = 0
    max_steps = args.max_steps if args.max_steps > 0 else float("inf")

    print("Training config:")
    print(f"  supervision_mode = {args.supervision_mode}")
    print(f"  freeze_coarse    = {args.freeze_coarse}")
    print(f"  train_latent     = {args.train_latent}")
    print(f"  main_loss        = {args.main_loss}")
    print(f"  lambda_final     = {args.lambda_final}")
    print(f"  lambda_residual  = {args.lambda_residual}")
    print(f"  lambda_coarse    = {args.lambda_coarse}")
    print(f"  lambda_surface   = {args.lambda_surface}")
    print(f"  lambda_sign      = {args.lambda_sign}")
    print(f"  lambda_eikonal   = {args.lambda_eikonal}")
    print(f"  lambda_normal    = {args.lambda_normal}")

    for epoch in range(1, args.num_epochs + 1):
        residual_model.train()

        if args.freeze_coarse == 1:
            coarse_model.eval()
        else:
            coarse_model.train()

        if args.train_latent == 1:
            latent_codebook.train()
        else:
            latent_codebook.eval()

        running_loss = 0.0
        valid_batches = 0

        for batch in dataloader:
            if global_step >= max_steps:
                break

            points = batch.get("sdf_points", None)
            sdf_gt = batch.get("sdf_gt", None)
            sample_indices = batch.get("sample_index", None)

            points, sdf_gt = ensure_shape_sdf(points, sdf_gt)

            if points is None or sdf_gt is None or sample_indices is None:
                continue

            if points.size(1) == 0:
                continue

            points = points.to(device)
            sdf_gt = sdf_gt.to(device)
            sample_indices = sample_indices.long().to(device)

            surface_points = batch.get("surface_points", None)
            surface_normals = batch.get("surface_normals", None)

            if isinstance(surface_points, torch.Tensor):
                surface_points = surface_points.to(device)

            if isinstance(surface_normals, torch.Tensor):
                surface_normals = surface_normals.to(device)

            z = latent_codebook(sample_indices).unsqueeze(1)

            coarse_pred = coarse_model(z, points)
            residual_pred = residual_model(z, points)
            final_pred = coarse_pred + residual_pred

            total_loss = torch.zeros(1, device=device, dtype=final_pred.dtype)

            loss_final = make_zero_like(final_pred)
            loss_residual = make_zero_like(final_pred)
            loss_coarse = make_zero_like(final_pred)
            loss_surface = make_zero_like(final_pred)
            loss_sign = make_zero_like(final_pred)
            loss_eikonal = make_zero_like(final_pred)
            loss_normal = make_zero_like(final_pred)

            # --------------------------------------------------
            # Main SDF supervision
            # --------------------------------------------------
            if args.supervision_mode in ["final", "both"]:
                loss_final = compute_regression_loss(
                    final_pred,
                    sdf_gt,
                    args.main_loss,
                )
                total_loss = total_loss + args.lambda_final * loss_final

            if args.supervision_mode in ["residual", "both"]:
                if args.detach_coarse_for_residual == 1:
                    residual_target = sdf_gt - coarse_pred.detach()
                else:
                    residual_target = sdf_gt - coarse_pred

                loss_residual = compute_regression_loss(
                    residual_pred,
                    residual_target,
                    args.main_loss,
                )
                total_loss = total_loss + args.lambda_residual * loss_residual

            # --------------------------------------------------
            # Optional coarse loss
            # Only useful when coarse model is trainable.
            # --------------------------------------------------
            if args.freeze_coarse == 0 and args.lambda_coarse > 0:
                loss_coarse = compute_regression_loss(
                    coarse_pred,
                    sdf_gt,
                    args.coarse_loss,
                )
                total_loss = total_loss + args.lambda_coarse * loss_coarse

            # --------------------------------------------------
            # Surface loss on final SDF
            # --------------------------------------------------
            if (
                args.use_surface_loss == 1
                and args.lambda_surface > 0
                and isinstance(surface_points, torch.Tensor)
                and surface_points.numel() > 0
            ):
                final_surface_pred = coarse_model(z, surface_points) + residual_model(
                    z,
                    surface_points,
                )
                loss_surface = final_surface_pred.abs().mean()
                total_loss = total_loss + args.lambda_surface * loss_surface

            # --------------------------------------------------
            # Sign loss on final SDF
            # --------------------------------------------------
            if args.use_sign_loss == 1 and args.lambda_sign > 0:
                sign_labels = torch.where(
                    sdf_gt >= 0,
                    torch.ones_like(sdf_gt),
                    -torch.ones_like(sdf_gt),
                )
                loss_sign = torch.relu(-sign_labels * final_pred).mean()
                total_loss = total_loss + args.lambda_sign * loss_sign

            # --------------------------------------------------
            # Eikonal loss on final SDF gradient
            # --------------------------------------------------
            if args.use_eikonal_loss == 1 and args.lambda_eikonal > 0:
                _, grads = predict_with_grad(
                    lambda latent, pts: coarse_model(latent, pts)
                    + residual_model(latent, pts),
                    z,
                    points,
                )
                grad_norm = torch.linalg.norm(grads, dim=-1)
                loss_eikonal = ((grad_norm - 1.0) ** 2).mean()
                total_loss = total_loss + args.lambda_eikonal * loss_eikonal

            # --------------------------------------------------
            # Normal consistency loss
            # --------------------------------------------------
            if (
                args.use_normal_loss == 1
                and args.lambda_normal > 0
                and isinstance(surface_points, torch.Tensor)
                and isinstance(surface_normals, torch.Tensor)
                and surface_points.numel() > 0
                and surface_normals.numel() > 0
            ):
                _, surface_grads = predict_with_grad(
                    lambda latent, pts: coarse_model(latent, pts)
                    + residual_model(latent, pts),
                    z,
                    surface_points,
                )

                n_pred = F.normalize(surface_grads, p=2, dim=-1, eps=1e-8)
                n_gt = F.normalize(surface_normals, p=2, dim=-1, eps=1e-8)

                loss_normal = (1.0 - (n_pred * n_gt).sum(dim=-1)).mean()
                total_loss = total_loss + args.lambda_normal * loss_normal

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.detach().item()
            valid_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                print(
                    f"epoch={epoch}/{args.num_epochs} "
                    f"step={global_step} "
                    f"loss={total_loss.detach().item():.6f} "
                    f"final={loss_final.detach().item():.6f} "
                    f"residual={loss_residual.detach().item():.6f} "
                    f"coarse={loss_coarse.detach().item():.6f} "
                    f"surface={loss_surface.detach().item():.6f} "
                    f"sign={loss_sign.detach().item():.6f} "
                    f"eikonal={loss_eikonal.detach().item():.6f} "
                    f"normal={loss_normal.detach().item():.6f}"
                )

            if wandb_run is not None and global_step % max(1, args.wandb_log_every) == 0:
                try:
                    wandb.log(
                        {
                            "train/loss": total_loss.detach().item(),
                            "train/loss_final": loss_final.detach().item(),
                            "train/loss_residual": loss_residual.detach().item(),
                            "train/loss_coarse": loss_coarse.detach().item(),
                            "train/loss_surface": loss_surface.detach().item(),
                            "train/loss_sign": loss_sign.detach().item(),
                            "train/loss_eikonal": loss_eikonal.detach().item(),
                            "train/loss_normal": loss_normal.detach().item(),
                            "train/epoch": epoch,
                            "train/step": global_step,
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

        if valid_batches == 0:
            print(
                f"epoch={epoch}: no valid SDF batches found. "
                "Ensure precomputed files exist under dataroot/sdf/<datalist>/*.npz"
            )
            continue

        epoch_loss = running_loss / float(valid_batches)
        print(
            f"epoch={epoch} "
            f"mean_loss={epoch_loss:.6f} "
            f"valid_batches={valid_batches}"
        )

        if wandb_run is not None:
            try:
                wandb.log(
                    {
                        "epoch/loss": epoch_loss,
                        "epoch": epoch,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )
            except Exception:
                pass

        checkpoint_out = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "coarse_state": coarse_model.state_dict(),
            "residual_state": residual_model.state_dict(),
            "latent_state": latent_codebook.state_dict(),
            "config": dict(vars(args)),
            "sample_names": list(sample_names),
        }

        latest_path = os.path.join(save_dir, "latest_net_DRMResidual.pth")
        torch.save(checkpoint_out, latest_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(save_dir, "best_net_DRMResidual.pth")
            torch.save(checkpoint_out, best_path)
            print(f"Updated best checkpoint: {best_path} mean_loss={best_loss:.6f}")

        if epoch % args.save_every == 0:
            epoch_path = os.path.join(save_dir, f"epoch_{epoch}_net_DRMResidual.pth")
            torch.save(checkpoint_out, epoch_path)

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