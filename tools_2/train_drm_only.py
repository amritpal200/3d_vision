
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 tools_2/train_drm_only.py --dataroot /data/113-1/users/asingh/project/3d/MPV3D --datalist train_pairs --datamode aligned --dataset_model MTM --name DRM_only_bootstrap --checkpoints_dir /data/113-1/users/asingh/project/3d/checkpoints --num_epochs 200 --batch_size 160 --sdf_hidden_dim 512 --sdf_num_layers 8 --pe_L 6

#!/usr/bin/env python3
"""Train a DRM-only SDF baseline without MTM.

The script learns:
1) A shared SDF MLP (same architecture used by DRM SDF branch), and
2) A per-sample latent codebook (one learnable latent vector per training sample).

This is intended as a quick starting point that can generate coarse human meshes.
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

sys.path.append('.')

from data import create_dataset
from models_2 import DRMSDFModel, LatentCodebook
from models_2.drm_only_model import build_checkpoint


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


def build_dataset_opt(args):
    opt = SimpleNamespace()
    opt.dataroot = args.dataroot
    opt.datalist = args.datalist
    opt.datamode = args.datamode
    # Use MTM data path branch by default to avoid hard dependency on warproot.
    opt.model = args.dataset_model
    opt.batch_size = args.batch_size
    opt.img_width = 320
    opt.img_height = 512
    opt.isTrain = True
    opt.max_dataset_size = float('inf')
    opt.num_threads = args.num_workers
    opt.serial_batches = False
    opt.no_pin_memory = False
    opt.radius = 5
    opt.warproot = args.warproot
    return opt


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def ensure_shape_sdf(points, sdf_gt):
    if points is None or sdf_gt is None:
        return None, None
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if sdf_gt.dim() == 2:
        sdf_gt = sdf_gt.unsqueeze(-1)
    if sdf_gt.dim() == 1:
        sdf_gt = sdf_gt.view(1, -1, 1)
    return points, sdf_gt


def predict_with_grad(model, latent_z, points):
    points_req = points.clone().detach().requires_grad_(True)
    sdf_pred = model(latent_z, points_req)
    sdf_sum = sdf_pred.sum()
    grads = torch.autograd.grad(
        outputs=sdf_sum,
        inputs=points_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return sdf_pred, grads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='mpv3d_example')
    parser.add_argument('--datalist', type=str, default='train_pairs')
    parser.add_argument('--datamode', type=str, default='aligned')
    parser.add_argument('--dataset_model', type=str, default='MTM', choices=['MTM', 'DRM'])
    parser.add_argument('--warproot', type=str, default='')
    parser.add_argument('--name', type=str, default='DRM_only_bootstrap')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--lr_model', type=float, default=1e-3)
    parser.add_argument('--lr_latent', type=float, default=3e-3)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--sdf_hidden_dim', type=int, default=512)
    parser.add_argument('--sdf_num_layers', type=int, default=8)
    parser.add_argument('--pe_L', type=int, default=6)
    parser.add_argument('--lambda_coarse', type=float, default=5.0)
    parser.add_argument('--lambda_surface', type=float, default=0.0)
    parser.add_argument('--lambda_sign', type=float, default=0.0)
    parser.add_argument('--lambda_eikonal', type=float, default=0.0)
    parser.add_argument('--lambda_normal', type=float, default=0.0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


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
            {'params': model.parameters(), 'lr': args.lr_model},
            {'params': latent_codebook.parameters(), 'lr': args.lr_latent},
        ],
        betas=(0.5, 0.999),
    )

    save_dir = os.path.join(args.checkpoints_dir, args.datamode, args.name)
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')
    global_step = 0
    max_steps = args.max_steps if args.max_steps > 0 else float('inf')

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        latent_codebook.train()
        running_loss = 0.0
        valid_batches = 0

        for batch in dataloader:
            if global_step >= max_steps:
                break

            points = batch.get('sdf_points', None)
            sdf_gt = batch.get('sdf_gt', None)
            sample_indices = batch.get('sample_index', None)
            points, sdf_gt = ensure_shape_sdf(points, sdf_gt)

            if points is None or sdf_gt is None or sample_indices is None:
                continue
            if points.size(1) == 0:
                continue

            points = points.to(device)
            sdf_gt = sdf_gt.to(device)
            sample_indices = sample_indices.long().to(device)

            surface_points = batch.get('surface_points', None)
            surface_normals = batch.get('surface_normals', None)
            sdf_scale = batch.get('sdf_scale', None)
            if isinstance(surface_points, torch.Tensor):
                surface_points = surface_points.to(device)
            if isinstance(surface_normals, torch.Tensor):
                surface_normals = surface_normals.to(device)
            if sdf_scale is None:
                sdf_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
            else:
                sdf_scale = sdf_scale.to(device)
                if sdf_scale.dim() == 0:
                    sdf_scale = sdf_scale.unsqueeze(0).expand(points.size(0))
                elif sdf_scale.dim() == 1 and sdf_scale.size(0) == 1:
                    sdf_scale = sdf_scale.expand(points.size(0))

            sign_labels = torch.where(sdf_gt >= 0, torch.ones_like(sdf_gt), -torch.ones_like(sdf_gt))

            z = latent_codebook(sample_indices).unsqueeze(1)

            total_loss = None

            if args.lambda_coarse > 0:
                sdf_pred = model(z, points)
                loss_coarse = F.mse_loss(sdf_pred, sdf_gt)
                total_loss = args.lambda_coarse * loss_coarse if total_loss is None else total_loss + args.lambda_coarse * loss_coarse
            else:
                sdf_pred = model(z, points)
                loss_coarse = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if args.lambda_surface > 0 and isinstance(surface_points, torch.Tensor) and surface_points.numel() > 0:
                surface_pred = model(z, surface_points)
                loss_surface = surface_pred.abs().mean()
                total_loss = args.lambda_surface * loss_surface if total_loss is None else total_loss + args.lambda_surface * loss_surface
            else:
                loss_surface = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if args.lambda_sign > 0:
                loss_sign = torch.relu(-sign_labels * sdf_pred).mean()
                total_loss = args.lambda_sign * loss_sign if total_loss is None else total_loss + args.lambda_sign * loss_sign
            else:
                loss_sign = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if args.lambda_eikonal > 0:
                _, grads = predict_with_grad(model, z, points)
                grad_norm = torch.linalg.norm(grads, dim=-1)
                target = 1.0 / sdf_scale
                if isinstance(target, torch.Tensor) and target.dim() == 1:
                    target = target.view(-1, 1)
                loss_eikonal = ((grad_norm - target) ** 2).mean()
                total_loss = args.lambda_eikonal * loss_eikonal if total_loss is None else total_loss + args.lambda_eikonal * loss_eikonal
            else:
                loss_eikonal = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if args.lambda_normal > 0 and isinstance(surface_points, torch.Tensor) and isinstance(surface_normals, torch.Tensor) and surface_points.numel() > 0 and surface_normals.numel() > 0:
                _, surface_grads = predict_with_grad(model, z, surface_points)
                n_pred = F.normalize(surface_grads, p=2, dim=-1, eps=1e-8)
                n_gt = F.normalize(surface_normals, p=2, dim=-1, eps=1e-8)
                loss_normal = (1.0 - (n_pred * n_gt).sum(dim=-1)).mean()
                total_loss = args.lambda_normal * loss_normal if total_loss is None else total_loss + args.lambda_normal * loss_normal
            else:
                loss_normal = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if total_loss is None:
                total_loss = sdf_pred.sum() * 0.0

            loss = total_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            valid_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                print(
                    f'epoch={epoch}/{args.num_epochs} step={global_step} '
                    f'loss={loss.detach().item():.6f} '
                    f'coarse={loss_coarse.detach().item():.6f} '
                    f'surface={loss_surface.detach().item():.6f} '
                    f'sign={loss_sign.detach().item():.6f} '
                    f'eikonal={loss_eikonal.detach().item():.6f} '
                    f'normal={loss_normal.detach().item():.6f}'
                )

        if valid_batches == 0:
            print(
                f'epoch={epoch}: no valid SDF batches found. '
                'Ensure precomputed files exist under dataroot/sdf/<datalist>/*.npz'
            )
            continue

        epoch_loss = running_loss / float(valid_batches)
        print(f'epoch={epoch} mean_loss={epoch_loss:.6f} valid_batches={valid_batches}')

        config = vars(args)
        checkpoint = build_checkpoint(
            model=model,
            latent_codebook=latent_codebook,
            epoch=epoch,
            global_step=global_step,
            config=config,
            sample_names=sample_names,
        )

        latest_path = os.path.join(save_dir, 'latest_net_DRMOnly.pth')
        torch.save(checkpoint, latest_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(save_dir, 'best_net_DRMOnly.pth')
            torch.save(checkpoint, best_path)
            print(f'Updated best checkpoint: {best_path} (mean_l1={best_loss:.6f})')

        if epoch % args.save_every == 0:
            epoch_path = os.path.join(save_dir, f'epoch_{epoch}_net_DRMOnly.pth')
            torch.save(checkpoint, epoch_path)

        if global_step >= max_steps:
            print(f'Reached --max_steps={args.max_steps}, stopping early.')
            break

    print('Training finished.')
    print(f'Checkpoints saved under: {save_dir}')


if __name__ == '__main__':
    main()
