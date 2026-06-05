
# !! Training
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 tools_2_image_encoder/train_drm_image_occupancy.py \
#   --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
#   --datalist train_pairs \
#   --datamode aligned \
#   --dataset_model MTM \
#   --name DRM_image_occupancy \
#   --checkpoints_dir /data/125-1/users/asingh/proves \
#   --num_epochs 200 \
#   --batch_size 10 \
#   --latent_dim 256 \
#   --sdf_hidden_dim 512 \
#   --sdf_num_layers 10 \
#   --pe_L 12 \
#   --lambda_surface 0.1 \
#   --lambda_logit_l2 1e-5



#!/usr/bin/env python3
"""Train image -> encoder -> latent z -> DRM-shaped occupancy field.

This is parallel to train_drm_image_conditioned.py, but the DRM MLP output is
interpreted as an occupancy logit instead of a signed distance value. Occupancy
targets are derived from the existing SDF supervision:

    occupancy = 1 when sdf_gt <= occupancy_threshold, else 0
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from common import (  # noqa: E402
    build_image_conditioned_checkpoint,
    build_models_from_args,
    create_image_sdf_dataset,
    get_device,
    load_training_checkpoint,
    prepare_sdf_batch,
    predict_with_grad,
    seed_everything,
    safe_collate,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", type=str, default="mpv3d_example")
    parser.add_argument("--datalist", type=str, default="train_pairs")
    parser.add_argument("--datamode", type=str, default="aligned")
    parser.add_argument("--dataset_model", type=str, default="MTM", choices=["MTM", "DRM"])
    parser.add_argument("--warproot", type=str, default="")
    parser.add_argument("--image_key", type=str, default="person")
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=5)

    parser.add_argument("--name", type=str, default="DRM_image_occupancy")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_optimizer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--pretrained_drm_checkpoint", type=str, default="")
    parser.add_argument("--freeze_drm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--strict_drm_load", type=int, default=1, choices=[0, 1])
    parser.add_argument("--auto_drm_arch_from_checkpoint", type=int, default=1, choices=[0, 1])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr_model", type=float, default=1e-4)
    parser.add_argument("--lr_encoder", type=float, default=3e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.5)
    parser.add_argument("--adam_beta2", type=float, default=0.999)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--sdf_hidden_dim", type=int, default=512)
    parser.add_argument("--sdf_num_layers", type=int, default=8)
    parser.add_argument("--pe_L", type=int, default=6)

    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--encoder_base_channels", type=int, default=32)
    parser.add_argument("--encoder_num_blocks", type=int, default=5)
    parser.add_argument("--encoder_head_hidden_dim", type=int, default=512)
    parser.add_argument("--encoder_dropout", type=float, default=0.0)
    parser.add_argument("--encoder_use_batchnorm", type=int, default=1, choices=[0, 1])

    parser.add_argument("--occupancy_threshold", type=float, default=0.0)
    parser.add_argument("--occupancy_loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--auto_pos_weight", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_pos_weight", type=float, default=20.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--lambda_occupancy", type=float, default=1.0)
    parser.add_argument(
        "--lambda_surface",
        type=float,
        default=0.1,
        help="Encourage surface points to lie on the occupancy decision boundary.",
    )
    parser.add_argument(
        "--lambda_eikonal",
        type=float,
        default=0.0,
        help="Optional gradient regularizer for a cleaner decision boundary.",
    )
    parser.add_argument(
        "--lambda_logit_l2",
        type=float,
        default=1e-5,
        help="Small logit penalty to discourage very noisy saturated fields.",
    )

    parser.add_argument("--wandb_project", type=str, default="m3d_drm")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


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


def extract_drm_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("occupancy_state", "drm_state", "model_state"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_pretrained_drm(drm, checkpoint_path, device, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = extract_drm_state(checkpoint)
    if hasattr(state, "_metadata"):
        del state._metadata
    result = drm.load_state_dict(state, strict=bool(strict))
    drm.to(device)
    print(f"Loaded pretrained DRM/occupancy field: {checkpoint_path}")
    if hasattr(result, "missing_keys") and result.missing_keys:
        print("DRM missing keys:", result.missing_keys[:20])
    if hasattr(result, "unexpected_keys") and result.unexpected_keys:
        print("DRM unexpected keys:", result.unexpected_keys[:20])
    return checkpoint


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = bool(trainable)
    module.train(bool(trainable))


def infer_drm_arch_from_state(state, preferred_pe_L=None, preferred_latent_dim=None):
    if "layers.0.weight" not in state:
        return None

    first_weight = state["layers.0.weight"]
    hidden_dim = int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])

    layer_ids = []
    for key in state.keys():
        if key.startswith("layers.") and key.endswith(".weight"):
            try:
                layer_ids.append(int(key.split(".")[1]))
            except Exception:
                pass
    num_layers = max(layer_ids) + 2 if layer_ids else 2

    candidates = []
    for pe_L in range(0, 32):
        encoded_point_dim = 3 * (1 + 2 * pe_L)
        latent_dim = input_dim - encoded_point_dim
        if latent_dim > 0:
            candidates.append((int(latent_dim), int(pe_L)))

    if not candidates:
        return None

    if preferred_latent_dim is not None:
        for latent_dim, pe_L in candidates:
            if latent_dim == int(preferred_latent_dim):
                return latent_dim, hidden_dim, num_layers, pe_L

    common_latents = (128, 256, 512, 1024, 64)
    for common_latent in common_latents:
        for latent_dim, pe_L in candidates:
            if latent_dim == common_latent:
                return latent_dim, hidden_dim, num_layers, pe_L

    if preferred_pe_L is not None:
        for latent_dim, pe_L in candidates:
            if pe_L == int(preferred_pe_L):
                return latent_dim, hidden_dim, num_layers, pe_L

    latent_dim, pe_L = candidates[-1]
    return latent_dim, hidden_dim, num_layers, pe_L


def maybe_update_arch_from_pretrained(args):
    if not args.pretrained_drm_checkpoint or not args.auto_drm_arch_from_checkpoint:
        return

    checkpoint = torch.load(args.pretrained_drm_checkpoint, map_location="cpu")
    state = extract_drm_state(checkpoint)
    inferred = infer_drm_arch_from_state(
        state,
        preferred_pe_L=args.pe_L,
        preferred_latent_dim=args.latent_dim,
    )
    if inferred is None:
        print("Could not infer DRM architecture from pretrained checkpoint; using CLI architecture.")
        return

    latent_dim, hidden_dim, num_layers, pe_L = inferred
    old = (args.latent_dim, args.sdf_hidden_dim, args.sdf_num_layers, args.pe_L)
    new = (latent_dim, hidden_dim, num_layers, pe_L)
    if old != new:
        print(
            "Auto-adjusting DRM architecture from pretrained checkpoint: "
            f"latent_dim {old[0]} -> {new[0]}, "
            f"sdf_hidden_dim {old[1]} -> {new[1]}, "
            f"sdf_num_layers {old[2]} -> {new[2]}, "
            f"pe_L {old[3]} -> {new[3]}"
        )
    args.latent_dim = latent_dim
    args.sdf_hidden_dim = hidden_dim
    args.sdf_num_layers = num_layers
    args.pe_L = pe_L


def checkpoint_paths(args):
    save_dir = os.path.join(args.checkpoints_dir, args.datamode, args.name)
    latest_path = os.path.join(save_dir, "latest_net_DRMOccupancy.pth")
    best_path = os.path.join(save_dir, "best_net_DRMOccupancy.pth")
    return save_dir, latest_path, best_path


def occupancy_targets_from_sdf(sdf_gt, threshold):
    return (sdf_gt <= float(threshold)).to(dtype=sdf_gt.dtype)


def bce_with_optional_pos_weight(logits, targets, args):
    pos_weight = None
    if args.auto_pos_weight:
        with torch.no_grad():
            positive = targets.sum()
            negative = targets.numel() - positive
            if positive.item() > 0:
                weight = negative / positive.clamp_min(1.0)
                weight = weight.clamp(min=1.0, max=float(args.max_pos_weight))
                pos_weight = weight.detach()

    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def focal_occupancy_loss(logits, targets, args):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = targets * prob + (1.0 - targets) * (1.0 - prob)
    alpha_t = targets * args.focal_alpha + (1.0 - targets) * (1.0 - args.focal_alpha)
    return (alpha_t * (1.0 - p_t).pow(args.focal_gamma) * bce).mean()


def compute_occupancy_losses(model, latent_z, batch, args):
    points = batch["points"]
    sdf_gt = batch["sdf_gt"]
    surface_points = batch["surface_points"]

    logits = model(latent_z, points)
    targets = occupancy_targets_from_sdf(sdf_gt, args.occupancy_threshold)

    if args.occupancy_loss == "focal":
        loss_occupancy = focal_occupancy_loss(logits, targets, args)
    else:
        loss_occupancy = bce_with_optional_pos_weight(logits, targets, args)

    total_loss = args.lambda_occupancy * loss_occupancy

    if args.lambda_surface > 0 and isinstance(surface_points, torch.Tensor) and surface_points.numel() > 0:
        surface_logits = model(latent_z, surface_points)
        loss_surface = surface_logits.abs().mean()
        total_loss = total_loss + args.lambda_surface * loss_surface
    else:
        loss_surface = torch.zeros(1, device=points.device, dtype=logits.dtype)

    if args.lambda_eikonal > 0:
        _, grads = predict_with_grad(model, latent_z, points)
        grad_norm = torch.linalg.norm(grads, dim=-1)
        loss_eikonal = ((grad_norm - 1.0) ** 2).mean()
        total_loss = total_loss + args.lambda_eikonal * loss_eikonal
    else:
        loss_eikonal = torch.zeros(1, device=points.device, dtype=logits.dtype)

    if args.lambda_logit_l2 > 0:
        loss_logit_l2 = logits.pow(2).mean()
        total_loss = total_loss + args.lambda_logit_l2 * loss_logit_l2
    else:
        loss_logit_l2 = torch.zeros(1, device=points.device, dtype=logits.dtype)

    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).to(dtype=targets.dtype)
        accuracy = (pred == targets).to(dtype=targets.dtype).mean()
        inside_ratio = targets.mean()

    return total_loss, {
        "occupancy": loss_occupancy,
        "surface": loss_surface,
        "eikonal": loss_eikonal,
        "logit_l2": loss_logit_l2,
        "accuracy": accuracy,
        "inside_ratio": inside_ratio,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.gpu_id)
    print(f"Using device: {device}")
    maybe_update_arch_from_pretrained(args)

    dataset, base_dataset = create_image_sdf_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check --dataroot and --datalist.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate,
    )
    sample_names = getattr(base_dataset, "im_names", [str(i) for i in range(len(dataset))])

    encoder, occupancy_model = build_models_from_args(args, device)
    if args.pretrained_drm_checkpoint:
        load_pretrained_drm(
            occupancy_model,
            checkpoint_path=args.pretrained_drm_checkpoint,
            device=device,
            strict=bool(args.strict_drm_load),
        )

    if args.freeze_drm:
        set_trainable(occupancy_model, False)
        print("Occupancy field is frozen; training only the image encoder.")
    else:
        set_trainable(occupancy_model, True)
        print("Occupancy field is trainable; training image encoder + field model.")

    optimizer_params = [{"params": encoder.parameters(), "lr": args.lr_encoder}]
    if not args.freeze_drm:
        optimizer_params.append({"params": occupancy_model.parameters(), "lr": args.lr_model})

    optimizer = torch.optim.Adam(optimizer_params, betas=(args.adam_beta1, args.adam_beta2))

    save_dir, latest_path, best_path = checkpoint_paths(args)
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    global_step = 0
    start_epoch = 1
    resume_path = args.resume_checkpoint
    if args.resume and not resume_path:
        resume_path = latest_path

    if resume_path:
        checkpoint = load_training_checkpoint(
            resume_path,
            encoder=encoder,
            drm=occupancy_model,
            optimizer=optimizer,
            device=device,
            load_optimizer=bool(args.resume_optimizer),
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", best_loss))
        print(f"Resumed from {resume_path} at epoch={start_epoch} global_step={global_step}")

    wandb_run = init_wandb(args)
    max_steps = args.max_steps if args.max_steps > 0 else float("inf")

    for epoch in range(start_epoch, args.num_epochs + 1):
        encoder.train()
        occupancy_model.train(not bool(args.freeze_drm))
        running_loss = 0.0
        valid_batches = 0

        for batch in dataloader:
            if global_step >= max_steps:
                break

            prepared = prepare_sdf_batch(batch, device)
            if prepared is None:
                continue

            latent_z = encoder(prepared["images"]).unsqueeze(1)
            loss, loss_terms = compute_occupancy_losses(occupancy_model, latent_z, prepared, args)

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
                    f"occ={loss_terms['occupancy'].detach().item():.6f} "
                    f"surface={loss_terms['surface'].detach().item():.6f} "
                    f"eikonal={loss_terms['eikonal'].detach().item():.6f} "
                    f"logit_l2={loss_terms['logit_l2'].detach().item():.6f} "
                    f"acc={loss_terms['accuracy'].detach().item():.4f} "
                    f"inside={loss_terms['inside_ratio'].detach().item():.4f}"
                )

            if wandb_run is not None and global_step % max(1, args.wandb_log_every) == 0:
                try:
                    wandb.log(
                        {
                            "train/loss": loss.detach().item(),
                            "train/loss_occupancy": loss_terms["occupancy"].detach().item(),
                            "train/loss_surface": loss_terms["surface"].detach().item(),
                            "train/loss_eikonal": loss_terms["eikonal"].detach().item(),
                            "train/loss_logit_l2": loss_terms["logit_l2"].detach().item(),
                            "train/occupancy_accuracy": loss_terms["accuracy"].detach().item(),
                            "train/inside_ratio": loss_terms["inside_ratio"].detach().item(),
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
        print(f"epoch={epoch} mean_loss={epoch_loss:.6f} valid_batches={valid_batches}")

        if wandb_run is not None:
            try:
                wandb.log({"epoch/loss": epoch_loss, "epoch": epoch}, step=global_step)
            except Exception:
                pass

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        config = dict(vars(args))
        config["field_type"] = "occupancy"
        config["iso_logit"] = 0.0
        config["iso_probability"] = 0.5
        checkpoint = build_image_conditioned_checkpoint(
            encoder=encoder,
            drm=occupancy_model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            config=config,
            sample_names=sample_names,
        )
        checkpoint["occupancy_state"] = checkpoint["drm_state"]
        torch.save(checkpoint, latest_path)

        if is_best:
            torch.save(checkpoint, best_path)
            print(f"Updated best checkpoint: {best_path} (mean_loss={best_loss:.6f})")

        # if epoch % args.save_every == 0:
        #     epoch_path = os.path.join(save_dir, f"epoch_{epoch}_net_DRMOccupancy.pth")
        #     torch.save(checkpoint, epoch_path)

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

