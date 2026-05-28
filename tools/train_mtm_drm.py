# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
# python3 tools/train_mtm_drm.py --num_epochs 100 --batch_size 100 --sdf_num_points 512 --save_freq 1000

"""Simple MTM+DRM trainer (proof-of-concept)

This script freezes a pretrained MTM, uses it to produce latent `z` from
simulated inputs, then trains the DRM (SDF MLP) for a few iterations.

Use this for fast integration testing only (no dataset wiring).
"""
import sys
import time
sys.path.append('.')
import argparse
import torch
from types import SimpleNamespace

from models import networks
from models.DRM_model import DRMModel
from data import create_dataset
import os
import wandb
from torch.utils.data import DataLoader, random_split

# Config
MTM_CKPT = '/data/113-1/users/asingh/project/3d/checkpoints/MTM/latest_net_MTM.pth'
NUM_ITERS = 20
BATCH_SIZE = 2

opt = SimpleNamespace()
opt.latent_dim = 128
opt.point_dim = 3
opt.sdf_hidden_dim = 128
opt.sdf_num_layers = 3
opt.sdf_num_points = 32
opt.lambda_coarse = 1.0
opt.lambda_surface = 0.1
opt.lambda_sign = 0.1
opt.lambda_eikonal = 0.1
opt.lambda_normal = 0.1
opt.norm = 'instance'
opt.init_type = 'normal'
opt.init_gain = 0.02
opt.gpu_ids = []
opt.isTrain = True
opt.lr = 0.001
opt.checkpoints_dir = '/data/113-1/users/asingh/project/3d/checkpoints/'
opt.datamode = 'aligned'
opt.name = 'DRM_train'
opt.display_ncols = 2
opt.ngf = 64
opt.save_freq = 10
opt.wandb_project = 'm3d_drm'
opt.num_epochs = 1
opt.val_frac = 0.1
opt.val_max_batches = 20
# use GPU automatically when available
opt.gpu_ids = [0] if torch.cuda.is_available() else []
# CLI overrides
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=opt.num_epochs)
parser.add_argument('--save_freq', type=int, default=opt.save_freq)
parser.add_argument('--val_frac', type=float, default=opt.val_frac)
parser.add_argument('--val_max_batches', type=int, default=opt.val_max_batches)
parser.add_argument('--sdf_num_points', type=int, default=opt.sdf_num_points)
parser.add_argument('--lambda_coarse', type=float, default=opt.lambda_coarse)
parser.add_argument('--lambda_surface', type=float, default=opt.lambda_surface)
parser.add_argument('--lambda_sign', type=float, default=opt.lambda_sign)
parser.add_argument('--lambda_eikonal', type=float, default=opt.lambda_eikonal)
parser.add_argument('--lambda_normal', type=float, default=opt.lambda_normal)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='training batch size')
parser.add_argument('--wandb_project', type=str, default=opt.wandb_project)
parser.add_argument('--max_iters', type=int, default=-1, help='If >0, cap total iterations; else run full epochs')
args = parser.parse_args()

# apply CLI overrides
opt.num_epochs = args.num_epochs
opt.save_freq = args.save_freq
opt.val_frac = args.val_frac
opt.val_max_batches = args.val_max_batches
opt.sdf_num_points = args.sdf_num_points
opt.lambda_coarse = args.lambda_coarse
opt.lambda_surface = args.lambda_surface
opt.lambda_sign = args.lambda_sign
opt.lambda_eikonal = args.lambda_eikonal
opt.lambda_normal = args.lambda_normal
opt.wandb_project = args.wandb_project
MAX_ITERS = args.max_iters if args.max_iters > 0 else float('inf')
opt.batch_size = args.batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else 'cpu')
if device.type == 'cuda':
    print(f'Using device: {device} ({torch.cuda.get_device_name(0)})')
else:
    print(f'Using device: {device}')

def main():
    # prepare options for dataset and models
    ds_opt = SimpleNamespace()
    ds_opt.dataroot = '/data/113-1/users/asingh/project/3d/MPV3D'
    ds_opt.datalist = 'train_pairs'
    ds_opt.datamode = 'aligned'
    ds_opt.model = 'MTM'
    ds_opt.batch_size = opt.batch_size
    ds_opt.img_width = 320
    ds_opt.img_height = 512
    ds_opt.isTrain = True
    ds_opt.max_dataset_size = float('inf')
    ds_opt.num_threads = 0
    ds_opt.serial_batches = False
    ds_opt.no_pin_memory = True
    ds_opt.radius = 5
    ds_opt.warproot = ''

    dataset_loader = create_dataset(ds_opt)
    full_dataset = dataset_loader.dataset

    # split into train/val
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * getattr(opt, 'val_frac', 0.1)))
    n_train = max(1, n_total - n_val)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_subset, batch_size=ds_opt.batch_size, shuffle=True,
                              num_workers=int(ds_opt.num_threads), pin_memory=not ds_opt.no_pin_memory)
    val_loader = DataLoader(val_subset, batch_size=ds_opt.batch_size, shuffle=False,
                            num_workers=int(ds_opt.num_threads), pin_memory=not ds_opt.no_pin_memory)

    # instantiate raw MTM network and load pretrained weights
    raw_mtm = networks.define_MTM(
        input_nc_A=29,
        input_nc_B=3,
        ngf=opt.ngf,
        n_layers=3,
        img_height=ds_opt.img_height,
        img_width=ds_opt.img_width,
        grid_size=3,
        add_tps=True,
        add_depth=True,
        add_segmt=True,
        latent_dim=opt.latent_dim,
        norm='instance',
        use_dropout=False,
        init_type='normal',
        init_gain=0.02,
        gpu_ids=opt.gpu_ids,
    )
    state = torch.load(MTM_CKPT, map_location='cpu')
    if hasattr(state, '_metadata'):
        del state._metadata
    # load checkpoint non-strict to allow missing keys (e.g., projection heads)
    load_res = raw_mtm.load_state_dict(state, strict=False)
    if hasattr(load_res, 'missing_keys') or hasattr(load_res, 'unexpected_keys'):
        print('MTM load results - missing keys:', getattr(load_res, 'missing_keys', None))
        print('MTM load results - unexpected keys:', getattr(load_res, 'unexpected_keys', None))
    raw_mtm.to(device)
    raw_mtm.eval()
    for p in raw_mtm.parameters():
        p.requires_grad = False

    # instantiate DRM model (training)
    drm = DRMModel(opt)
    drm.train()

    # initialize wandb
    try:
        wandb.init(project=opt.wandb_project, name=opt.name, config=vars(opt))
    except Exception:
        print('wandb init failed or offline; continuing without remote logging')

    start = time.time()
    it = 0
    best_val = float('inf')
    for epoch in range(getattr(opt, 'num_epochs', 1)):
        print(f'=== Epoch {epoch+1}/{opt.num_epochs} ===')
        for i, data in enumerate(train_loader):
            if it >= MAX_ITERS:
                break

            # feed data to raw MTM network
            agnostic = data.get('agnostic', None)
            cloth = data.get('cloth', None)
            if isinstance(agnostic, torch.Tensor):
                agnostic = agnostic.to(device)
            if isinstance(cloth, torch.Tensor):
                cloth = cloth.to(device)
            with torch.no_grad():
                mtm_out = raw_mtm(agnostic, cloth)
            z = mtm_out.get('z')
            if z is not None:
                z = z.to(drm.device)

            # prepare DRM batch using precomputed SDF fields from the dataset
            drm_batch = dict(data)
            for k, v in list(drm_batch.items()):
                if isinstance(v, torch.Tensor):
                    drm_batch[k] = v.to(drm.device)
            if z is not None:
                drm_batch['z'] = z

            drm.set_input(drm_batch)
            drm.optimize_parameters()

            it += 1
            # log training loss to console and wandb
            loss_val = drm.loss_sdf.detach().item()
            if it % 5 == 0:
                print(f'Iter {it}/{MAX_ITERS}  loss_sdf={loss_val:.6f}')
            try:
                wandb.log({'train/loss_sdf': loss_val, 'train/iter': it, 'train/epoch': epoch+1})
            except Exception:
                pass

            # small validation using the same batch (no grad)
            if it % 500 == 0:
                drm.eval()
                with torch.no_grad():
                    drm.forward()
                    val_loss = torch.abs(drm.sdf_pred - drm.sdf_gt).mean().item()
                print(f'  Val (mini) loss: {val_loss:.6f}')
                try:
                    wandb.log({'val/loss_sdf': val_loss, 'val/iter': it, 'val/epoch': epoch+1})
                except Exception:
                    pass
                drm.train()

            # periodic checkpoint save
            if it % opt.save_freq == 0:
                os.makedirs(drm.save_dir, exist_ok=True)
                drm.save_networks(it)
                print(f'Saved checkpoint at iter {it} to {drm.save_dir}')

        # end of epoch: save per-epoch checkpoint
        os.makedirs(drm.save_dir, exist_ok=True)
        drm.save_networks(f'epoch_{epoch+1}')
        epoch_ckpt = os.path.join(drm.save_dir, f'epoch_{epoch+1}_net_DRM.pth')
        print(f'End of epoch {epoch+1}: saved {epoch_ckpt}')
        # run validation over val_loader (limited batches)
        drm.eval()
        val_losses = []
        max_batches = getattr(opt, 'val_max_batches', 20)
        with torch.no_grad():
            for vi, vdata in enumerate(val_loader):
                if vi >= max_batches:
                    break
                vbatch = dict(vdata)
                for k, v in list(vbatch.items()):
                    if isinstance(v, torch.Tensor):
                        vbatch[k] = v.to(drm.device)
                drm.set_input(vbatch)
                drm.forward()
                val_losses.append(torch.abs(drm.sdf_pred - drm.sdf_gt).mean().item())
        drm.train()
        if val_losses:
            mean_val = float(sum(val_losses) / len(val_losses))
            print(f'Validation mean loss: {mean_val:.6f}')
            try:
                wandb.log({'val/epoch_loss': mean_val, 'epoch': epoch+1})
            except Exception:
                pass
            # save best model by validation loss (local only)
            if mean_val < best_val:
                best_val = mean_val
                print(f'New best val {best_val:.6f} -> saving best model')
                os.makedirs(drm.save_dir, exist_ok=True)
                drm.save_networks('best')

    # final save
    os.makedirs(drm.save_dir, exist_ok=True)
    drm.save_networks('final')
    final_ckpt = os.path.join(drm.save_dir, f'final_net_DRM.pth')
    # do NOT upload checkpoints to wandb; only metrics are logged online
    print('Done. Time: %.2fs' % (time.time() - start))

if __name__ == '__main__':
    main()
