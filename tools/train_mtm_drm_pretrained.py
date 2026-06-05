#!/usr/bin/env python3
"""Train a pretrained DRM SDF model with MTM-produced latent z.

This is the MTM-driven counterpart to ``tools/train_mtm_drm.py``.
It loads:
1) a pretrained DRM SDF checkpoint from ``tools_2/train_drm_only.py``, and
2) a pretrained MTM checkpoint that emits latent ``z`` through ``z_proj``.

Training updates the DRM SDF weights and the MTM latent projection head
jointly, so the latent code remains trainable while still coming from MTM.
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


def load_state_dict_flex(module, checkpoint_path, device, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state', checkpoint)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    load_result = module.load_state_dict(state_dict, strict=strict)
    module.to(device)
    return checkpoint, load_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='mpv3d_example')
    parser.add_argument('--datalist', type=str, default='train_pairs')
    parser.add_argument('--datamode', type=str, default='aligned')
    parser.add_argument('--dataset_model', type=str, default='MTM', choices=['MTM', 'DRM'])
    parser.add_argument('--warproot', type=str, default='')
    parser.add_argument('--name', type=str, default='MTM_DRM_pretrained')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--pretrained_drm_checkpoint', type=str, required=True)
    parser.add_argument('--pretrained_mtm_checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--lr_drm', type=float, default=1e-4)
    parser.add_argument('--lr_z_proj', type=float, default=3e-4)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--sdf_hidden_dim', type=int, default=512)
    parser.add_argument('--sdf_num_layers', type=int, default=8)
    parser.add_argument('--pe_L', type=int, default=6)
    parser.add_argument('--lambda_coarse', type=float, default=5.0)
    parser.add_argument('--lambda_surface', type=float, default=0.1)
    parser.add_argument('--lambda_sign', type=float, default=0.1)
    parser.add_argument('--lambda_eikonal', type=float, default=0.1)
    parser.add_argument('--lambda_normal', type=float, default=0.1)
    parser.add_argument('--mtm_input_nc_A', type=int, default=29)
    parser.add_argument('--mtm_input_nc_B', type=int, default=3)
    parser.add_argument('--mtm_ngf', type=int, default=64)
    parser.add_argument('--mtm_n_layers_feat_extract', type=int, default=3)
    parser.add_argument('--mtm_grid_size', type=int, default=3)
    parser.add_argument('--mtm_add_tps', action='store_true', default=True)
    parser.add_argument('--mtm_add_depth', action='store_true', default=True)
    parser.add_argument('--mtm_add_segmt', action='store_true', default=True)
    parser.add_argument('--mtm_norm', type=str, default='instance')
    parser.add_argument('--mtm_use_dropout', action='store_true', default=False)
    parser.add_argument('--mtm_init_type', type=str, default='normal')
    parser.add_argument('--mtm_init_gain', type=float, default=0.02)
    parser.add_argument('--wandb_project', type=str, default='m3d_drm')
    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_log_every', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def init_wandb(args, config):
    if wandb is None or args.wandb_mode == 'disabled':
        print('wandb not available or disabled; continuing without remote logging')
        return None

    run_name = args.wandb_name.strip() or args.name
    try:
        return wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            mode=args.wandb_mode,
        )
    except Exception:
        print('wandb init failed; continuing without remote logging')
        return None


def infer_drm_hyperparams(args, drm_checkpoint):
    config = drm_checkpoint.get('config', {}) if isinstance(drm_checkpoint, dict) else {}
    latent_dim = int(config.get('latent_dim', args.latent_dim))
    sdf_hidden_dim = int(config.get('sdf_hidden_dim', args.sdf_hidden_dim))
    sdf_num_layers = int(config.get('sdf_num_layers', args.sdf_num_layers))
    pe_L = int(config.get('pe_L', args.pe_L))
    return latent_dim, sdf_hidden_dim, sdf_num_layers, pe_L


def build_mtm_model(args, latent_dim, device):
    model = networks.define_MTM(
        input_nc_A=args.mtm_input_nc_A,
        input_nc_B=args.mtm_input_nc_B,
        ngf=args.mtm_ngf,
        n_layers=args.mtm_n_layers_feat_extract,
        img_height=512,
        img_width=320,
        grid_size=args.mtm_grid_size,
        add_tps=args.mtm_add_tps,
        add_depth=args.mtm_add_depth,
        add_segmt=args.mtm_add_segmt,
        latent_dim=latent_dim,
        norm=args.mtm_norm,
        use_dropout=args.mtm_use_dropout,
        init_type=args.mtm_init_type,
        init_gain=args.mtm_init_gain,
        gpu_ids=[device.index] if device.type == 'cuda' else [],
    )
    return model


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.gpu_id)
    print(f'Using device: {device}')

    drm_checkpoint = torch.load(args.pretrained_drm_checkpoint, map_location='cpu')
    latent_dim, sdf_hidden_dim, sdf_num_layers, pe_L = infer_drm_hyperparams(args, drm_checkpoint)

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

    drm_model = DRMSDFModel(
        latent_dim=latent_dim,
        point_dim=3,
        hidden_dim=sdf_hidden_dim,
        num_layers=sdf_num_layers,
        pe_L=pe_L,
    ).to(device)
    drm_state_dict = drm_checkpoint.get('model_state', drm_checkpoint)
    if hasattr(drm_state_dict, '_metadata'):
        del drm_state_dict._metadata
    drm_model.load_state_dict(drm_state_dict)

    mtm_model = build_mtm_model(args, latent_dim, device)
    mtm_checkpoint, mtm_load_result = load_state_dict_flex(mtm_model, args.pretrained_mtm_checkpoint, device, strict=False)
    if hasattr(mtm_load_result, 'unexpected_keys') and getattr(mtm_load_result, 'unexpected_keys', None):
        print('MTM load unexpected keys:', getattr(mtm_load_result, 'unexpected_keys', None))
    if hasattr(mtm_load_result, 'missing_keys') and getattr(mtm_load_result, 'missing_keys', None):
        print('MTM load missing keys:', getattr(mtm_load_result, 'missing_keys', None))

    drm_model.train()
    mtm_model.eval()
    for param in mtm_model.parameters():
        param.requires_grad = False
    if hasattr(mtm_model, 'z_proj'):
        for param in mtm_model.z_proj.parameters():
            param.requires_grad = True
        print('MTM z_proj will be trained jointly with DRM.')
    else:
        raise RuntimeError('MTM checkpoint/model does not expose z_proj; cannot train latent z jointly.')

    optimizer_drm = torch.optim.Adam(
        drm_model.parameters(),
        lr=args.lr_drm,
        betas=(0.5, 0.999),
    )
    optimizer_z = torch.optim.Adam(
        mtm_model.z_proj.parameters(),
        lr=args.lr_z_proj,
        betas=(0.5, 0.999),
    )

    save_dir = os.path.join(args.checkpoints_dir, args.datamode, args.name)
    os.makedirs(save_dir, exist_ok=True)

    wandb_run = init_wandb(args, vars(args))

    best_loss = float('inf')
    global_step = 0
    max_steps = args.max_steps if args.max_steps > 0 else float('inf')

    for epoch in range(1, args.num_epochs + 1):
        drm_model.train()
        mtm_model.eval()
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
            if points.size(1) == 0:
                continue

            agnostic = batch.get('agnostic', None)
            cloth = batch.get('cloth', None)
            if not isinstance(agnostic, torch.Tensor) or not isinstance(cloth, torch.Tensor):
                raise RuntimeError('MTM training requires dataset fields "agnostic" and "cloth".')

            agnostic = agnostic.to(device)
            cloth = cloth.to(device)
            points = points.to(device)
            sdf_gt = sdf_gt.to(device)

            surface_points = batch.get('surface_points', None)
            surface_normals = batch.get('surface_normals', None)
            if isinstance(surface_points, torch.Tensor):
                surface_points = surface_points.to(device)
            if isinstance(surface_normals, torch.Tensor):
                surface_normals = surface_normals.to(device)

            sign_labels = torch.where(sdf_gt >= 0, torch.ones_like(sdf_gt), -torch.ones_like(sdf_gt))

            optimizer_drm.zero_grad(set_to_none=True)
            optimizer_z.zero_grad(set_to_none=True)

            mtm_out = mtm_model(agnostic, cloth)
            z = mtm_out.get('z')
            if z is None:
                raise RuntimeError('MTM forward did not return a latent z.')

            sdf_pred = drm_model(z, points)

            total_loss = None
            if args.lambda_coarse > 0:
                loss_coarse = F.mse_loss(sdf_pred, sdf_gt)
                total_loss = args.lambda_coarse * loss_coarse
            else:
                loss_coarse = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if args.lambda_surface > 0 and isinstance(surface_points, torch.Tensor) and surface_points.numel() > 0:
                surface_pred = drm_model(z, surface_points)
                loss_surface = surface_pred.abs().mean()
                total_loss = loss_surface * args.lambda_surface if total_loss is None else total_loss + args.lambda_surface * loss_surface
            else:
                loss_surface = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if args.lambda_sign > 0:
                loss_sign = torch.relu(-sign_labels * sdf_pred).mean()
                total_loss = loss_sign * args.lambda_sign if total_loss is None else total_loss + args.lambda_sign * loss_sign
            else:
                loss_sign = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if args.lambda_eikonal > 0:
                _, grads = predict_with_grad(drm_model, z, points)
                grad_norm = torch.linalg.norm(grads, dim=-1)
                loss_eikonal = ((grad_norm - 1.0) ** 2).mean()
                total_loss = loss_eikonal * args.lambda_eikonal if total_loss is None else total_loss + args.lambda_eikonal * loss_eikonal
            else:
                loss_eikonal = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if args.lambda_normal > 0 and isinstance(surface_points, torch.Tensor) and isinstance(surface_normals, torch.Tensor) and surface_points.numel() > 0 and surface_normals.numel() > 0:
                _, surface_grads = predict_with_grad(drm_model, z, surface_points)
                n_pred = F.normalize(surface_grads, p=2, dim=-1, eps=1e-8)
                n_gt = F.normalize(surface_normals, p=2, dim=-1, eps=1e-8)
                loss_normal = (1.0 - (n_pred * n_gt).sum(dim=-1)).mean()
                total_loss = loss_normal * args.lambda_normal if total_loss is None else total_loss + args.lambda_normal * loss_normal
            else:
                loss_normal = torch.zeros(1, device=device, dtype=sdf_pred.dtype)

            if total_loss is None:
                total_loss = sdf_pred.sum() * 0.0

            total_loss.backward()
            optimizer_drm.step()
            optimizer_z.step()

            running_loss += total_loss.detach().item()
            valid_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                print(
                    f'epoch={epoch}/{args.num_epochs} step={global_step} '
                    f'loss={total_loss.detach().item():.6f} '
                    f'coarse={loss_coarse.detach().item():.6f} '
                    f'surface={loss_surface.detach().item():.6f} '
                    f'sign={loss_sign.detach().item():.6f} '
                    f'eikonal={loss_eikonal.detach().item():.6f} '
                    f'normal={loss_normal.detach().item():.6f}'
                )

            if wandb_run is not None and global_step % max(1, args.wandb_log_every) == 0:
                try:
                    wandb.log(
                        {
                            'train/loss': total_loss.detach().item(),
                            'train/loss_coarse': loss_coarse.detach().item(),
                            'train/loss_surface': loss_surface.detach().item(),
                            'train/loss_sign': loss_sign.detach().item(),
                            'train/loss_eikonal': loss_eikonal.detach().item(),
                            'train/loss_normal': loss_normal.detach().item(),
                            'train/epoch': epoch,
                            'train/step': global_step,
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

        if valid_batches == 0:
            print(
                f'epoch={epoch}: no valid SDF batches found. '
                'Ensure the dataset provides "sdf_points" and "sdf_gt".'
            )
            continue

        epoch_loss = running_loss / float(valid_batches)
        print(f'epoch={epoch} mean_loss={epoch_loss:.6f} valid_batches={valid_batches}')

        if wandb_run is not None:
            try:
                wandb.log(
                    {
                        'epoch/loss': epoch_loss,
                        'epoch': epoch,
                        'train/global_step': global_step,
                    },
                    step=global_step,
                )
            except Exception:
                pass

        checkpoint = {
            'epoch': int(epoch),
            'global_step': int(global_step),
            'model_state': drm_model.state_dict(),
            'drm_state': drm_model.state_dict(),
            'mtm_state': mtm_model.state_dict(),
            'config': dict(vars(args)),
            'sample_names': list(sample_names),
        }

        latest_path = os.path.join(save_dir, 'latest_net_MTM_DRM.pth')
        torch.save(checkpoint, latest_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(save_dir, 'best_net_MTM_DRM.pth')
            torch.save(checkpoint, best_path)
            print(f'Updated best checkpoint: {best_path} (mean_loss={best_loss:.6f})')

        if epoch % args.save_every == 0:
            epoch_path = os.path.join(save_dir, f'epoch_{epoch}_net_MTM_DRM.pth')
            torch.save(checkpoint, epoch_path)

        if global_step >= max_steps:
            print(f'Reached --max_steps={args.max_steps}, stopping early.')
            break

    print('Training finished.')
    print(f'Checkpoints saved under: {save_dir}')

    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()