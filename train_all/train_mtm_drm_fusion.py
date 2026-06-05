
# Scenario 1
# Train only new fusion layers + image encoder
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
# python3 train_all/train_mtm_drm_fusion.py \
# --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
# --datalist train_pairs \
# --datamode aligned \
# --mtm_ckpt /data/113-1/users/asingh/project/3d/checkpoints/MTM/latest_net_MTM.pth \
# --pretrained_coarse_checkpoint /data/113-1/users/asingh/project/3d/checkpoints/5th_version/aligned/DRM_only_bootstrap/epoch_32_net_DRMOnly.pth \
# --checkpoints_dir /data/113-1/users/asingh/project/3d/checkpoints/fusion_person_only \
# --name fusion_only_new_layers \
# --batch_size 8 \
# --num_epochs 200 \
# --gpu_id 0 \
# --drm_mode coarse \
# --freeze_mtm 1 \
# --freeze_coarse 1 \
# --latent_dim 128 \
# --mtm_z_dim 1024 \
# --sdf_hidden_dim 812 \
# --sdf_num_layers 10 \
# --pe_L 12 \
# --image_in_channels 3 \
# --image_feature_dim 256 \
# --image_scale 0.1 \
# --fusion_scale 0.1

# Scenario 2
# Train image encoder + fusion + coarse DRM
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
# python3 train_all/train_mtm_drm_fusion.py \
# --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
# --datalist train_pairs \
# --datamode aligned \
# --mtm_ckpt /data/113-1/users/asingh/project/3d/checkpoints/MTM/latest_net_MTM.pth \
# --pretrained_coarse_checkpoint /data/113-1/users/asingh/project/3d/checkpoints/5th_version/aligned/DRM_only_bootstrap/epoch_32_net_DRMOnly.pth \
# --checkpoints_dir /data/113-1/users/asingh/project/3d/checkpoints/fusion_person_only \
# --name fusion_train_coarse \
# --batch_size 8 \
# --num_epochs 200 \
# --gpu_id 0 \
# --drm_mode coarse \
# --freeze_mtm 1 \
# --freeze_coarse 0 \
# --latent_dim 128 \
# --mtm_z_dim 1024 \
# --sdf_hidden_dim 812 \
# --sdf_num_layers 10 \
# --pe_L 12 \
# --image_in_channels 3 \
# --image_feature_dim 256 \
# --image_scale 0.1 \
# --fusion_scale 0.1


# Scenario 3
# Train everything
# Train:
# MTM, image encoder, z_proj, fusion_mlp,DRM coarse
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
# python3 train_all/train_mtm_drm_fusion.py \
# --dataroot /data/113-1/users/asingh/project/3d/MPV3D \
# --datalist train_pairs \
# --datamode aligned \
# --mtm_ckpt /data/113-1/users/asingh/project/3d/checkpoints/MTM/latest_net_MTM.pth \
# --pretrained_coarse_checkpoint /data/113-1/users/asingh/project/3d/checkpoints/5th_version/aligned/DRM_only_bootstrap/epoch_32_net_DRMOnly.pth \
# --checkpoints_dir /data/113-1/users/asingh/project/3d/checkpoints/fusion_person_only \
# --name fusion_train_all \
# --batch_size 8 \
# --num_epochs 200 \
# --gpu_id 0 \
# --drm_mode coarse \
# --freeze_mtm 0 \
# --freeze_coarse 0 \
# --latent_dim 128 \
# --mtm_z_dim 1024 \
# --sdf_hidden_dim 812 \
# --sdf_num_layers 10 \
# --pe_L 12 \
# --image_in_channels 3 \
# --image_feature_dim 256 \
# --image_scale 0.1 \
# --fusion_scale 0.1

#!/usr/bin/env python3

"""Train an image-conditioned DRM + residual SDF model using MTM-produced z.

This script reuses the pretrained MTM branch to produce latent z, adds an
image encoder over person+agnostic inputs, and trains a DRM coarse+residual
SDF on the fused latent.
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

sys.path.append('.')

from data import create_dataset
from models import networks
from models_2 import DRMSDFModel
from train_all.fusion_sdf_model import ImageConditionedFusionSDF


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample['sample_index'] = idx
        return sample


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def build_dataset_opt(args):
    opt = SimpleNamespace()
    opt.dataroot = args.dataroot
    opt.datalist = args.datalist
    opt.datamode = args.datamode
    opt.model = 'MTM'
    opt.batch_size = args.batch_size
    opt.img_width = args.img_width
    opt.img_height = args.img_height
    opt.isTrain = True
    opt.max_dataset_size = float('inf')
    opt.num_threads = args.num_workers
    opt.serial_batches = False
    opt.no_pin_memory = False
    opt.radius = args.radius
    opt.warproot = ''
    return opt


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


def predict_with_grad(model_fn, latent_z, image_tensor, points):
    points_req = points.clone().detach().requires_grad_(True)
    outputs = model_fn(latent_z, image_tensor, points_req)
    sdf_pred = outputs['final_sdf']
    grads = torch.autograd.grad(
        outputs=sdf_pred.sum(),
        inputs=points_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return sdf_pred, grads


def load_state_dict(path):
    state = torch.load(path, map_location='cpu')
    if hasattr(state, '_metadata'):
        del state._metadata
    return state


def load_mtm(ckpt_path, latent_dim, device, args):
    net = networks.define_MTM(
        input_nc_A=args.mtm_input_nc_A,
        input_nc_B=args.mtm_input_nc_B,
        ngf=args.mtm_ngf,
        n_layers=args.mtm_n_layers_feat_extract,
        img_height=args.img_height,
        img_width=args.img_width,
        grid_size=args.mtm_grid_size,
        add_tps=True,
        add_depth=True,
        add_segmt=True,
        latent_dim=latent_dim,
        norm=args.mtm_norm,
        use_dropout=args.mtm_use_dropout,
        init_type=args.mtm_init_type,
        init_gain=args.mtm_init_gain,
        gpu_ids=[device.index] if device.type == 'cuda' else [],
    )
    ckpt = load_state_dict(ckpt_path)
    state = ckpt.get('mtm_state') or ckpt.get('model_state') or ckpt
    if hasattr(state, '_metadata'):
        del state._metadata
    net.load_state_dict(state, strict=False)
    net.to(device)
    return net


def load_coarse_drm(ckpt_path, latent_dim, sdf_hidden_dim, sdf_num_layers, pe_L, device):
    net = DRMSDFModel(
        latent_dim=latent_dim,
        point_dim=3,
        hidden_dim=sdf_hidden_dim,
        num_layers=sdf_num_layers,
        pe_L=pe_L,
    )
    ckpt = load_state_dict(ckpt_path)
    state = ckpt.get('model_state') or ckpt.get('coarse_state') or ckpt.get('drm_state') or ckpt
    if hasattr(state, '_metadata'):
        del state._metadata
    net.load_state_dict(state, strict=False)
    net.to(device)
    return net


def load_residual_drm(ckpt_path, latent_dim, sdf_hidden_dim, sdf_num_layers, pe_L, device):
    net = DRMSDFModel(
        latent_dim=latent_dim,
        point_dim=3,
        hidden_dim=sdf_hidden_dim,
        num_layers=sdf_num_layers,
        pe_L=pe_L,
    )
    ckpt = load_state_dict(ckpt_path)
    state = ckpt.get('model_state') or ckpt.get('residual_state') or ckpt.get('drm_state') or ckpt
    if hasattr(state, '_metadata'):
        del state._metadata
    net.load_state_dict(state, strict=False)
    net.to(device)
    return net


def init_wandb(args):
    if wandb is None or args.wandb_mode == 'disabled':
        print('wandb not available or disabled; continuing without remote logging')
        return None
    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name.strip() or args.name,
            config=vars(args),
            mode=args.wandb_mode,
        )
    except Exception:
        print('wandb init failed; continuing without remote logging')
        return None


def save_checkpoint(model, mtm, epoch, global_step, args, sample_names, path):
    checkpoint = {
        'epoch': int(epoch),
        'global_step': int(global_step),
        'config': dict(vars(args)),
        'sample_names': list(sample_names),
        'drm_mode': args.drm_mode,
        'image_fusion_state': model.state_dict(),
        'coarse_state': model.coarse.state_dict(),
        'image_encoder_state': model.image_encoder.state_dict(),
        'z_proj_state': model.z_proj.state_dict(),
        'fusion_mlp_state': model.fusion_mlp.state_dict(),
        'mtm_state': mtm.state_dict(),
    }
    if model.residual is not None:
        checkpoint['residual_state'] = model.residual.state_dict()
    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='mpv3d_example')
    parser.add_argument('--datalist', type=str, default='train_pairs')
    parser.add_argument('--datamode', type=str, default='aligned')
    parser.add_argument('--name', type=str, default='train_all_fusion')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)

    parser.add_argument('--mtm_ckpt', type=str, required=True)
    parser.add_argument('--pretrained_coarse_checkpoint', type=str, required=True)
    parser.add_argument('--pretrained_residual_checkpoint', type=str, default='', help='optional pretrained residual DRM checkpoint')
    parser.add_argument('--drm_mode', type=str, default='residual', choices=['coarse', 'residual'])

    parser.add_argument('--freeze_mtm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--freeze_coarse', type=int, default=1, choices=[0, 1])

    parser.add_argument('--img_width', type=int, default=320)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--radius', type=int, default=5)

    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--sdf_hidden_dim', type=int, default=512)
    parser.add_argument('--sdf_num_layers', type=int, default=8)
    parser.add_argument('--pe_L', type=int, default=6)

    parser.add_argument('--mtm_input_nc_A', type=int, default=29)
    parser.add_argument('--mtm_input_nc_B', type=int, default=3)
    parser.add_argument('--mtm_ngf', type=int, default=64)
    parser.add_argument('--mtm_n_layers_feat_extract', type=int, default=3)
    parser.add_argument('--mtm_grid_size', type=int, default=3)
    parser.add_argument('--mtm_norm', type=str, default='instance')
    parser.add_argument('--mtm_use_dropout', action='store_true', default=False)
    parser.add_argument('--mtm_init_type', type=str, default='normal')
    parser.add_argument('--mtm_init_gain', type=float, default=0.02)

    parser.add_argument('--image_in_channels', type=int, default=3)
    parser.add_argument('--image_feature_dim', type=int, default=256)
    parser.add_argument('--image_scale', type=float, default=0.1)
    parser.add_argument('--residual_scale', type=float, default=1.0)
    parser.add_argument('--mtm_z_dim', type=int, default=1024)
    parser.add_argument('--fusion_scale', type=float, default=1.0)

    parser.add_argument('--lr_model', type=float, default=1e-4)
    parser.add_argument('--lr_mtm', type=float, default=1e-5)
    parser.add_argument('--lr_latent', type=float, default=3e-4)
    parser.add_argument('--lr_image_encoder', type=float, default=1e-4)

    parser.add_argument('--lambda_coarse', type=float, default=2.0)
    parser.add_argument('--lambda_surface', type=float, default=0.1)
    parser.add_argument('--lambda_sign', type=float, default=0.1)
    parser.add_argument('--lambda_eikonal', type=float, default=0.1)
    parser.add_argument('--lambda_normal', type=float, default=0.1)

    parser.add_argument('--main_loss', type=str, default='l2', choices=['l1', 'l2', 'smoothl1'])
    parser.add_argument('--wandb_project', type=str, default='m3d_drm')
    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_log_every', type=int, default=1)
    return parser.parse_args()


def regression_loss(pred, target, loss_type):
    if loss_type == 'l1':
        return F.l1_loss(pred, target)
    if loss_type == 'l2':
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.gpu_id)
    print(f'Using device: {device}')

    ds_opt = build_dataset_opt(args)
    dataset_loader = create_dataset(ds_opt)
    base_dataset = dataset_loader.dataset
    dataset = IndexedDataset(base_dataset)
    if len(dataset) == 0:
        raise RuntimeError('Dataset is empty. Check --dataroot and --datalist.')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    sample_names = getattr(base_dataset, 'im_names', [str(i) for i in range(len(dataset))])

    mtm = load_mtm(args.mtm_ckpt, args.mtm_z_dim, device, args)
    fusion = ImageConditionedFusionSDF(
        latent_dim=args.latent_dim,
        mtm_z_dim=args.mtm_z_dim,
        sdf_hidden_dim=args.sdf_hidden_dim,
        sdf_num_layers=args.sdf_num_layers,
        pe_L=args.pe_L,
        image_in_channels=3,
        image_feature_dim=args.image_feature_dim,
        image_scale=args.image_scale,
        fusion_scale=args.fusion_scale,
        residual_scale=args.residual_scale,
        drm_mode=args.drm_mode,
    ).to(device)
    fusion.coarse = load_coarse_drm(
        args.pretrained_coarse_checkpoint,
        args.latent_dim,
        args.sdf_hidden_dim,
        args.sdf_num_layers,
        args.pe_L,
        device,
    )
    if args.drm_mode == 'residual' and not args.pretrained_residual_checkpoint:
        raise RuntimeError('--pretrained_residual_checkpoint is required when --drm_mode=residual')
    if args.drm_mode == 'residual' and args.pretrained_residual_checkpoint:
        fusion.residual = load_residual_drm(
            args.pretrained_residual_checkpoint,
            args.latent_dim,
            args.sdf_hidden_dim,
            args.sdf_num_layers,
            args.pe_L,
            device,
        )
    else:
        fusion.residual = None

    print(f'DRM mode: {args.drm_mode}')
    if fusion.residual is not None:
        print('Residual branch enabled')
    else:
        print('Residual branch disabled')

    if args.freeze_coarse == 1:
        for param in fusion.coarse.parameters():
            param.requires_grad = False
    if args.freeze_mtm == 1:
        for param in mtm.parameters():
            param.requires_grad = False

    optimizer_groups = []
    if fusion.residual is not None and any(param.requires_grad for param in fusion.residual.parameters()):
        optimizer_groups.append({'params': fusion.residual.parameters(), 'lr': args.lr_model})
    if any(param.requires_grad for param in fusion.image_encoder.parameters()):
        optimizer_groups.append({'params': fusion.image_encoder.parameters(), 'lr': args.lr_image_encoder})
    
    # if any(param.requires_grad for param in fusion.image_proj.parameters()):
    #     optimizer_groups.append({'params': fusion.image_proj.parameters(), 'lr': args.lr_image_encoder})

    if any(p.requires_grad for p in fusion.z_proj.parameters()):
        optimizer_groups.append({
            'params': fusion.z_proj.parameters(),
            'lr': args.lr_latent
        })

    if any(p.requires_grad for p in fusion.fusion_mlp.parameters()):
        optimizer_groups.append({
            'params': fusion.fusion_mlp.parameters(),
            'lr': args.lr_latent
        })
    
    if any(param.requires_grad for param in fusion.coarse.parameters()):
        optimizer_groups.append({'params': fusion.coarse.parameters(), 'lr': args.lr_model})
    if any(param.requires_grad for param in mtm.parameters()):
        optimizer_groups.append({'params': mtm.parameters(), 'lr': args.lr_mtm})

    optimizer = torch.optim.Adam(optimizer_groups, betas=(0.5, 0.999))

    save_dir = os.path.join(args.checkpoints_dir, args.datamode, args.name)
    os.makedirs(save_dir, exist_ok=True)
    wandb_run = init_wandb(args)

    global_step = 0
    max_steps = args.max_steps if args.max_steps > 0 else float('inf')

    for epoch in range(1, args.num_epochs + 1):
        fusion.train()
        mtm.train(not args.freeze_mtm)
        running_loss = 0.0
        valid_batches = 0

        for batch in dataloader:
            if global_step >= max_steps:
                break

            points = batch.get('sdf_points', None)
            sdf_gt = batch.get('sdf_gt', None)
            points, sdf_gt = ensure_shape_sdf(points, sdf_gt)
            if points is None or sdf_gt is None:
                continue

            points = points.to(device)
            sdf_gt = sdf_gt.to(device)

            surface_points = batch.get('surface_points', None)
            surface_normals = batch.get('surface_normals', None)
            if isinstance(surface_points, torch.Tensor):
                if surface_points.dim() == 2:
                    surface_points = surface_points.unsqueeze(0)
                surface_points = surface_points.to(device)
            if isinstance(surface_normals, torch.Tensor):
                if surface_normals.dim() == 2:
                    surface_normals = surface_normals.unsqueeze(0)
                surface_normals = surface_normals.to(device)

            if 'agnostic' not in batch or 'person' not in batch or 'cloth' not in batch:
                continue

            agnostic = batch['agnostic'].to(device)
            person = batch['person'].to(device)
            cloth = batch['cloth'].to(device)
            image_tensor = person

            mtm_out = mtm(agnostic, cloth)
            latent_z = mtm_out['z']
            if latent_z.dim() == 3 and latent_z.size(1) == 1:
                latent_z = latent_z.squeeze(1)

            outputs = fusion(latent_z, image_tensor, points)
            final_pred = outputs['final_sdf']


            if global_step % 200 == 0:
                print("=" * 60)
                print("DEBUG LATENT STATS")
                print("=" * 60)

                print("z_projected mean:",
                    outputs["z_projected"].mean().item())
                print("z_projected std:",
                    outputs["z_projected"].std().item())

                print("image_feature mean:",
                    outputs["image_feature"].mean().item())
                print("image_feature std:",
                    outputs["image_feature"].std().item())

                print("latent_delta mean:",
                    outputs["latent_delta"].mean().item())
                print("latent_delta std:",
                    outputs["latent_delta"].std().item())

                print("fused_z mean:",
                    outputs["fused_z"].mean().item())
                print("fused_z std:",
                    outputs["fused_z"].std().item())



            sign_labels = torch.where(sdf_gt >= 0, torch.ones_like(sdf_gt), -torch.ones_like(sdf_gt))
            total_loss = fusion.final_sdf.sum() * 0.0 if hasattr(fusion, 'final_sdf') else final_pred.sum() * 0.0

            loss_coarse = regression_loss(final_pred, sdf_gt, args.main_loss)
            total_loss = total_loss + args.lambda_coarse * loss_coarse

            if surface_points is not None and surface_points.numel() > 0:
                surface_out = fusion(latent_z, image_tensor, surface_points)['final_sdf']
                loss_surface = surface_out.abs().mean()
                total_loss = total_loss + args.lambda_surface * loss_surface
            else:
                loss_surface = torch.zeros(1, device=device, dtype=final_pred.dtype)

            loss_sign = torch.relu(-sign_labels * final_pred).mean()
            total_loss = total_loss + args.lambda_sign * loss_sign

            _, grads = predict_with_grad(fusion, latent_z, image_tensor, points)
            grad_norm = torch.linalg.norm(grads, dim=-1)
            loss_eikonal = ((grad_norm - 1.0) ** 2).mean()
            total_loss = total_loss + args.lambda_eikonal * loss_eikonal

            if surface_points is not None and surface_normals is not None and surface_points.numel() > 0 and surface_normals.numel() > 0:
                surface_points_req = surface_points.clone().detach().requires_grad_(True)
                surface_pred = fusion(latent_z, image_tensor, surface_points_req)['final_sdf']
                surface_grads = torch.autograd.grad(
                    outputs=surface_pred.sum(),
                    inputs=surface_points_req,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                n_pred = F.normalize(surface_grads, p=2, dim=-1, eps=1e-8)
                n_gt = F.normalize(surface_normals, p=2, dim=-1, eps=1e-8)
                loss_normal = (1.0 - (n_pred * n_gt).sum(dim=-1)).mean()
                total_loss = total_loss + args.lambda_normal * loss_normal
            else:
                loss_normal = torch.zeros(1, device=device, dtype=final_pred.dtype)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.detach().item())
            valid_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                print(
                    f'epoch={epoch}/{args.num_epochs} step={global_step} '
                    f'loss={float(total_loss.detach().item()):.6f} '
                    f'coarse={float(loss_coarse.detach().item()):.6f} '
                    f'surface={float(loss_surface.detach().item()):.6f} '
                    f'sign={float(loss_sign.detach().item()):.6f} '
                    f'eikonal={float(loss_eikonal.detach().item()):.6f} '
                    f'normal={float(loss_normal.detach().item()):.6f}'
                )

            if wandb_run is not None and global_step % max(1, args.wandb_log_every) == 0:
                try:
                    wandb.log({
                        'loss': float(total_loss.detach().item()),
                        'loss_coarse': float(loss_coarse.detach().item()),
                        'loss_surface': float(loss_surface.detach().item()),
                        'loss_sign': float(loss_sign.detach().item()),
                        'loss_eikonal': float(loss_eikonal.detach().item()),
                        'loss_normal': float(loss_normal.detach().item()),
                        'epoch': epoch,
                        'step': global_step,
                    })
                except Exception:
                    pass

        avg_loss = running_loss / max(1, valid_batches)
        print(f'epoch {epoch} average loss: {avg_loss:.6f}')

        if epoch % args.save_every == 0:
            # ckpt_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
            # save_checkpoint(fusion, mtm, epoch, global_step, args, sample_names, ckpt_path)
            # print(f'Saved checkpoint: {ckpt_path}')
            # save as latest checkpoint to save memory, instead of saving separate checkpoint for each epoch
            ckpt_path = os.path.join(save_dir, 'latest_net_train_all_fusion.pth')
            save_checkpoint(fusion, mtm, epoch, global_step, args, sample_names, ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')


    final_path = os.path.join(save_dir, 'best_net_train_all_fusion.pth')
    save_checkpoint(fusion, mtm, args.num_epochs, global_step, args, sample_names, final_path)
    print(f'Saved final checkpoint: {final_path}')


if __name__ == '__main__':
    main()
