#!/usr/bin/env python3
"""Train only MTM z_proj as an extra latent input for image-conditioned DRM.

Frozen by default:
    - MTM backbone
    - image encoder
    - DRM field

Trainable:
    - MTM z_proj only

The frozen image encoder produces z_image. MTM z_proj produces z_mtm. The DRM
receives a fused latent, by default z_image + z_mtm, plus 3D query points.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
IMAGE_ENCODER_DIR = os.path.join(PROJECT_ROOT, "tools_2_image_encoder")
for path in (PROJECT_ROOT, CURRENT_DIR, IMAGE_ENCODER_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from tools_2_mtm_z.common import (  # noqa: E402
    build_checkpoint,
    build_concat_drm_from_loaded_drm,
    build_mtm,
    create_mtm_dataset,
    freeze_module,
    get_device,
    load_image_drm_checkpoint,
    load_mtm_pretrained,
    prepare_mtm_inputs,
    safe_collate,
    seed_everything,
    train_only_mtm_z_projection,
)
from tools_2_image_encoder.common import compute_drm_sdf_losses, load_training_checkpoint, prepare_sdf_batch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="mpv3d_example")
    parser.add_argument("--datalist", type=str, default="train_pairs")
    parser.add_argument("--datamode", type=str, default="aligned")
    parser.add_argument("--warproot", type=str, default="")
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=5)

    parser.add_argument("--image_drm_checkpoint", type=str, required=True)
    parser.add_argument("--pretrained_mtm_checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, default="MTM_z_for_image_DRM")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_optimizer", type=int, default=1, choices=[0, 1])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr_z_proj", type=float, default=3e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_drm", type=float, default=1e-5)
    parser.add_argument("--lr_mtm", type=float, default=1e-5)
    parser.add_argument("--train_mode", type=str, default="mtm_z_only", choices=["mtm_z_only", "all"])
    parser.add_argument("--adam_beta1", type=float, default=0.5)
    parser.add_argument("--adam_beta2", type=float, default=0.999)

    parser.add_argument("--lambda_coarse", type=float, default=2.0)
    parser.add_argument("--lambda_surface", type=float, default=0.1)
    parser.add_argument("--lambda_sign", type=float, default=0.1)
    parser.add_argument("--lambda_eikonal", type=float, default=0.0)
    parser.add_argument("--lambda_normal", type=float, default=0.0)
    parser.add_argument("--mtm_fusion_mode", type=str, default="add", choices=["add", "replace", "concat"])
    parser.add_argument("--mtm_z_scale", type=float, default=1.0)
    parser.add_argument("--zero_init_z_proj", type=int, default=1, choices=[0, 1])

    parser.add_argument("--mtm_input_nc_A", type=int, default=29)
    parser.add_argument("--mtm_input_nc_B", type=int, default=3)
    parser.add_argument("--mtm_ngf", type=int, default=64)
    parser.add_argument("--mtm_n_layers_feat_extract", type=int, default=3)
    parser.add_argument("--mtm_grid_size", type=int, default=3)
    parser.add_argument("--mtm_add_tps", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mtm_add_depth", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mtm_add_segmt", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mtm_norm", type=str, default="instance")
    parser.add_argument("--mtm_use_dropout", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mtm_init_type", type=str, default="normal")
    parser.add_argument("--mtm_init_gain", type=float, default=0.02)

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
    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name.strip() or args.name,
            config=dict(vars(args)),
            mode=args.wandb_mode,
        )
    except Exception as exc:
        print(f"wandb init failed; continuing without remote logging: {exc}")
        return None


def fuse_latents(z_image, z_mtm, args):
    scaled_mtm = args.mtm_z_scale * z_mtm
    if args.mtm_fusion_mode == "replace":
        return scaled_mtm
    if args.mtm_fusion_mode == "concat":
        return torch.cat([z_image, scaled_mtm], dim=-1)
    return z_image + scaled_mtm


def set_train_mode(mtm, encoder, drm, args):
    if args.train_mode == "all":
        for module in (mtm, encoder, drm):
            for param in module.parameters():
                param.requires_grad = True
            module.train()
        print("Training MTM, image encoder, and DRM with separate low-LR parameter groups.")
        return

    train_only_mtm_z_projection(mtm)
    freeze_module(encoder)
    freeze_module(drm)
    print("Training only mtm.z_proj; MTM backbone, image encoder, and DRM are frozen.")


def build_optimizer(mtm, encoder, drm, args):
    if args.train_mode == "all":
        return torch.optim.Adam(
            [
                {"params": mtm.parameters(), "lr": args.lr_mtm},
                {"params": encoder.parameters(), "lr": args.lr_encoder},
                {"params": drm.parameters(), "lr": args.lr_drm},
            ],
            betas=(args.adam_beta1, args.adam_beta2),
        )
    return torch.optim.Adam(
        mtm.z_proj.parameters(),
        lr=args.lr_z_proj,
        betas=(args.adam_beta1, args.adam_beta2),
    )


def maybe_zero_init_z_proj(mtm, args):
    if not args.zero_init_z_proj:
        return
    with torch.no_grad():
        mtm.z_proj.weight.zero_()
        if mtm.z_proj.bias is not None:
            mtm.z_proj.bias.zero_()
    print("Initialized mtm.z_proj to zero, so training starts from image encoder z only.")


def checkpoint_paths(args):
    save_dir = os.path.join(args.checkpoints_dir, args.datamode, args.name)
    return (
        save_dir,
        os.path.join(save_dir, "latest_net_MTMZImageDRM.pth"),
        os.path.join(save_dir, "best_net_MTMZImageDRM.pth"),
    )


def load_resume(path, mtm, encoder, drm, optimizer, device, load_optimizer=True):
    checkpoint = torch.load(path, map_location="cpu")
    mtm.load_state_dict(checkpoint["mtm_state"], strict=False)
    if "mtm_z_proj_state" in checkpoint:
        mtm.z_proj.load_state_dict(checkpoint["mtm_z_proj_state"])
    encoder.load_state_dict(checkpoint["encoder_state"])
    drm.load_state_dict(checkpoint["drm_state"])
    if load_optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    return checkpoint


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.gpu_id)
    print(f"Using device: {device}")

    image_ckpt, runtime, encoder, loaded_drm = load_image_drm_checkpoint(args.image_drm_checkpoint, device)
    image_latent_dim = int(runtime.latent_dim)
    mtm_latent_dim = image_latent_dim
    if args.mtm_fusion_mode == "concat":
        drm = build_concat_drm_from_loaded_drm(loaded_drm, runtime, device)
        drm_latent_dim = image_latent_dim + mtm_latent_dim
    else:
        drm = loaded_drm
        drm_latent_dim = image_latent_dim
    print("Loaded image encoder + DRM checkpoint.")
    print(
        f"Latent/DRM config: image_latent_dim={image_latent_dim}, mtm_latent_dim={mtm_latent_dim}, "
        f"drm_latent_dim={drm_latent_dim}, hidden={runtime.sdf_hidden_dim}, "
        f"layers={runtime.sdf_num_layers}, pe_L={runtime.pe_L}"
    )

    mtm = build_mtm(args, latent_dim=mtm_latent_dim, device=device)
    load_mtm_pretrained(mtm, args.pretrained_mtm_checkpoint, device)
    maybe_zero_init_z_proj(mtm, args)
    set_train_mode(mtm, encoder, drm, args)
    optimizer = build_optimizer(mtm, encoder, drm, args)

    save_dir, latest_path, best_path = checkpoint_paths(args)
    os.makedirs(save_dir, exist_ok=True)

    resume_path = args.resume_checkpoint
    if args.resume and not resume_path:
        resume_path = latest_path
    best_loss = float("inf")
    global_step = 0
    start_epoch = 1
    if resume_path:
        resume_ckpt = load_resume(
            resume_path,
            mtm=mtm,
            encoder=encoder,
            drm=drm,
            optimizer=optimizer,
            device=device,
            load_optimizer=bool(args.resume_optimizer),
        )
        set_train_mode(mtm, encoder, drm, args)
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_loss = float(resume_ckpt.get("best_loss", best_loss))
        print(f"Resumed from {resume_path} at epoch={start_epoch} global_step={global_step}")

    dataset, base_dataset = create_mtm_dataset(args, is_train=True, serial_batches=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate,
    )
    sample_names = getattr(base_dataset, "im_names", [str(i) for i in range(len(dataset))])
    wandb_run = init_wandb(args)
    max_steps = args.max_steps if args.max_steps > 0 else float("inf")

    for epoch in range(start_epoch, args.num_epochs + 1):
        set_train_mode(mtm, encoder, drm, args)
        running_loss = 0.0
        valid_batches = 0

        for batch in dataloader:
            if global_step >= max_steps:
                break
            prepared = prepare_sdf_batch(batch, device)
            mtm_inputs = prepare_mtm_inputs(batch, device)
            if prepared is None or mtm_inputs is None:
                continue

            agnostic, cloth = mtm_inputs
            mtm_output = mtm(agnostic, cloth)
            z_mtm = mtm_output.get("z", None)
            if z_mtm is None:
                raise RuntimeError("MTM forward did not return output['z'].")

            if args.train_mode == "all":
                z_image = encoder(prepared["images"]).unsqueeze(1)
            else:
                with torch.no_grad():
                    z_image = encoder(prepared["images"]).unsqueeze(1)
            fused_z = fuse_latents(z_image, z_mtm, args)
            loss, loss_terms = compute_drm_sdf_losses(drm, fused_z, prepared, args)
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
                    f"coarse={loss_terms['coarse'].detach().item():.6f} "
                    f"surface={loss_terms['surface'].detach().item():.6f} "
                    f"sign={loss_terms['sign'].detach().item():.6f}"
                )

            if wandb_run is not None and global_step % max(1, args.wandb_log_every) == 0:
                try:
                    wandb.log(
                        {
                            "train/loss": loss.detach().item(),
                            "train/loss_coarse": loss_terms["coarse"].detach().item(),
                            "train/loss_surface": loss_terms["surface"].detach().item(),
                            "train/loss_sign": loss_terms["sign"].detach().item(),
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

        if valid_batches == 0:
            print(f"epoch={epoch}: no valid batches")
            continue

        epoch_loss = running_loss / float(valid_batches)
        print(f"epoch={epoch} mean_loss={epoch_loss:.6f} valid_batches={valid_batches}")
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        config = dict(vars(args))
        config.update(
            {
                "latent_dim": drm_latent_dim,
                "image_latent_dim": image_latent_dim,
                "mtm_latent_dim": mtm_latent_dim,
                "drm_latent_dim": drm_latent_dim,
                "sdf_hidden_dim": runtime.sdf_hidden_dim,
                "sdf_num_layers": runtime.sdf_num_layers,
                "pe_L": runtime.pe_L,
                "image_drm_config": image_ckpt.get("config", {}),
            }
        )
        checkpoint = build_checkpoint(
            mtm=mtm,
            encoder=encoder,
            drm=drm,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            config=config,
            sample_names=sample_names,
        )
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, best_path)
            print(f"Updated best checkpoint: {best_path} (mean_loss={best_loss:.6f})")
        # if epoch % args.save_every == 0:
        #     torch.save(checkpoint, os.path.join(save_dir, f"epoch_{epoch}_net_MTMZImageDRM.pth"))
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

