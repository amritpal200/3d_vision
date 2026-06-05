
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 tools_2/reconstruct_drm_residual_mesh.py --checkpoint /data/113-1/users/asingh/project/3d/checkpoints/residual_corase_train_residual/aligned/DRM_residual_final_full/best_net_DRMResidual.pth --data_dir /data/113-1/users/asingh/project/3d/MPV3D --sample_index 0 --output_obj mesh_results/recon_residual.obj --resolution 96 --chunk_size 65536 --gpu_id 0 --save_point_cloud

#!/usr/bin/env python3
"""Reconstruct mesh from a residual DRM checkpoint (coarse + residual + latent).

Loads `coarse_state`, `residual_state`, and `latent_state` from a checkpoint
and evaluates final SDF = coarse(z,p) + residual(z,p) on a grid, then runs
marching cubes and writes OBJ (and optional PLY).
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append('.')

from models_2 import DRMSDFModel, LatentCodebook
from data import create_dataset

try:
    from skimage import measure
except Exception as exc:
    raise ImportError('scikit-image is required for marching cubes. Install with "pip install scikit-image".') from exc


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def select_sample_index(sample_names, sample_index, sample_name):
    if sample_name:
        if sample_name not in sample_names:
            raise ValueError(f'Sample name {sample_name} not found in checkpoint sample list.')
        return sample_names.index(sample_name)
    if sample_index < 0 or sample_index >= len(sample_names):
        raise IndexError(f'--sample_index must be in [0, {len(sample_names)-1}]')
    return sample_index


def sample_bounds_from_dataset(sample, padding=0.10):
    surface = sample.get('surface_points', None)
    if isinstance(surface, torch.Tensor) and surface.numel() > 0:
        pts = surface.detach().cpu().numpy()
    else:
        pts = sample.get('sdf_points', None)
        if isinstance(pts, torch.Tensor) and pts.numel() > 0:
            pts = pts.detach().cpu().numpy()
        else:
            pts = None

    if pts is not None and pts.shape[-1] == 3 and pts.shape[0] > 0:
        mins = pts.min(axis=0) - padding
        maxs = pts.max(axis=0) + padding
        return (
            (float(mins[0]), float(maxs[0])),
            (float(mins[1]), float(maxs[1])),
            (float(mins[2]), float(maxs[2])),
        )
    return (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)


def sample_bounds_from_sdf_npz(data_dir, datalist, sample_name, padding=0.10):
    if not data_dir or not datalist or not sample_name:
        return None

    sdf_path = os.path.join(data_dir, 'sdf', datalist, sample_name.replace('.png', '.npz'))
    if not os.path.exists(sdf_path):
        return None

    try:
        sdf_npz = np.load(sdf_path)
    except Exception:
        return None

    pts = None
    if 'surface_points' in sdf_npz and sdf_npz['surface_points'].size > 0:
        pts = sdf_npz['surface_points']
    elif 'points' in sdf_npz and sdf_npz['points'].size > 0:
        pts = sdf_npz['points']

    if pts is None or pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None

    mins = pts.min(axis=0) - padding
    maxs = pts.max(axis=0) + padding
    return (
        (float(mins[0]), float(maxs[0])),
        (float(mins[1]), float(maxs[1])),
        (float(mins[2]), float(maxs[2])),
    )


def evaluate_sdf_grid(model_fn, z, x_bounds, y_bounds, z_bounds, resolution, chunk_size, device):
    xs = np.linspace(x_bounds[0], x_bounds[1], resolution, dtype=np.float32)
    ys = np.linspace(y_bounds[0], y_bounds[1], resolution, dtype=np.float32)
    zs = np.linspace(z_bounds[0], z_bounds[1], resolution, dtype=np.float32)

    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing='ij')
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    sdf_values = []
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk_size):
            end = min(start + chunk_size, points.shape[0])
            p = torch.from_numpy(points[start:end]).float().to(device).unsqueeze(0)
            pred = model_fn(z, p)
            sdf_values.append(pred.squeeze(0).squeeze(-1).detach().cpu().numpy())

    sdf = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)
    return sdf, xs, ys, zs


def save_obj(path, vertices_xyz, faces):
    with open(path, 'w') as f:
        for v in vertices_xyz:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for tri in faces:
            f.write(f'f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n')


def save_point_cloud_ply(points, normals, output_path):
    has_normals = normals is not None and len(normals) == len(points)
    with open(output_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if has_normals:
            f.write('property float nx\n')
            f.write('property float ny\n')
            f.write('property float nz\n')
        f.write('end_header\n')
        if has_normals:
            for p, n in zip(points, normals):
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n')
        else:
            for p in points:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='mpv3d_example')
    parser.add_argument('--datalist', type=str, default='train_pairs')
    parser.add_argument('--datamode', type=str, default='aligned')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--warproot', type=str, default='')
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--sample_name', type=str, default='')
    parser.add_argument('--output_obj', type=str, default='mesh_results/recon_residual.obj')
    parser.add_argument('--resolution', type=int, default=96)
    parser.add_argument('--chunk_size', type=int, default=65536)
    parser.add_argument('--iso_level', type=float, default=0.0)
    parser.add_argument('--padding', type=float, default=0.10)
    parser.add_argument('--no_dataset_bounds', action='store_true')
    parser.add_argument('--save_point_cloud', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--sdf_hidden_dim', type=int, default=None)
    parser.add_argument('--sdf_num_layers', type=int, default=None)
    parser.add_argument('--pe_L', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    print(f'Using device: {device}')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    sample_names = checkpoint.get('sample_names', [])
    if sample_names is None:
        sample_names = []

    latent_dim = int(config.get('latent_dim', 128))
    sdf_hidden_dim = int(config.get('sdf_hidden_dim', 512))
    sdf_num_layers = int(config.get('sdf_num_layers', 8))
    pe_L = int(config.get('pe_L', 6))

    # CLI overrides
    if args.latent_dim is not None:
        latent_dim = int(args.latent_dim)
    if args.sdf_hidden_dim is not None:
        sdf_hidden_dim = int(args.sdf_hidden_dim)
    if args.sdf_num_layers is not None:
        sdf_num_layers = int(args.sdf_num_layers)
    if args.pe_L is not None:
        pe_L = int(args.pe_L)

    # Build models
    coarse = DRMSDFModel(latent_dim=latent_dim, point_dim=3, hidden_dim=sdf_hidden_dim, num_layers=sdf_num_layers, pe_L=pe_L).to(device)
    residual = DRMSDFModel(latent_dim=latent_dim, point_dim=3, hidden_dim=sdf_hidden_dim, num_layers=sdf_num_layers, pe_L=pe_L).to(device)
    latent = LatentCodebook(num_embeddings=max(1, len(sample_names)), latent_dim=latent_dim).to(device)

    # Load states (flexible)
    ck = checkpoint
    # coarse_state
    coarse_state = ck.get('coarse_state') or ck.get('drm_state') or ck.get('model_state') or ck
    try:
        if hasattr(coarse_state, '_metadata'):
            del coarse_state._metadata
    except Exception:
        pass
    try:
        coarse.load_state_dict(coarse_state)
    except Exception:
        coarse.load_state_dict(coarse_state, strict=False)

    # residual_state
    residual_state = ck.get('residual_state') or ck.get('residual_model_state') or None
    if residual_state is not None:
        try:
            if hasattr(residual_state, '_metadata'):
                del residual_state._metadata
        except Exception:
            pass
        try:
            residual.load_state_dict(residual_state)
        except Exception:
            residual.load_state_dict(residual_state, strict=False)
    else:
        print('Warning: checkpoint contains no residual_state; residual model left with random init.')

    # latent_state
    latent_state = ck.get('latent_state') or ck.get('latent') or None
    if latent_state is not None:
        try:
            latent.load_state_dict(latent_state)
            print('Loaded latent codebook from checkpoint.')
        except Exception as exc:
            print(f'Could not load latent_state: {exc}; proceeding with random init.')
    else:
        print('No latent_state found in checkpoint; latent will be randomly initialized.')

    coarse.eval()
    residual.eval()
    latent.eval()

    if not args.no_dataset_bounds and args.data_dir:
        try:
            ds_opt = SimpleNamespace()
            ds_opt.dataroot = args.data_dir
            ds_opt.datalist = args.datalist
            ds_opt.datamode = args.datamode
            ds_opt.model = 'DRM'
            ds_opt.batch_size = 1
            ds_opt.img_width = 320
            ds_opt.img_height = 512
            ds_opt.isTrain = False
            ds_opt.max_dataset_size = float('inf')
            ds_opt.num_threads = 0
            ds_opt.serial_batches = True
            ds_opt.no_pin_memory = True
            ds_opt.radius = 5
            ds_opt.warproot = args.warproot or args.data_dir
            dataset = create_dataset(ds_opt).dataset
            if len(dataset) == 0:
                print('Dataset empty; using default cube bounds.')
                x_bounds, y_bounds, z_bounds = (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)
            else:
                idx = select_sample_index(sample_names or [str(i) for i in range(len(dataset))], args.sample_index, args.sample_name)
                sample = dataset[idx]
                x_bounds, y_bounds, z_bounds = sample_bounds_from_dataset(sample, padding=args.padding)
        except Exception as exc:
            print('Failed to infer bounds from dataset. Reason:', exc)

            # Fallback: infer bounds directly from precomputed SDF .npz to avoid full-cube reconstruction.
            sample_name_for_bounds = ''
            if args.sample_name:
                sample_name_for_bounds = args.sample_name
            elif len(sample_names) > 0 and 0 <= args.sample_index < len(sample_names):
                sample_name_for_bounds = sample_names[args.sample_index]

            npz_bounds = sample_bounds_from_sdf_npz(
                data_dir=args.data_dir,
                datalist=args.datalist,
                sample_name=sample_name_for_bounds,
                padding=args.padding,
            )
            if npz_bounds is not None:
                x_bounds, y_bounds, z_bounds = npz_bounds
                print(f'Using bounds from SDF NPZ for sample: {sample_name_for_bounds}')
            else:
                print('SDF NPZ bounds unavailable; using default cube.')
                x_bounds, y_bounds, z_bounds = (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)
    else:
        x_bounds, y_bounds, z_bounds = (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)

    # prepare z
    if len(sample_names) == 0:
        # if sample names missing, assume at least one
        sample_names = [str(i) for i in range(latent.num_embeddings)]

    idx = select_sample_index(sample_names, args.sample_index, args.sample_name)
    z = latent(torch.tensor([idx], dtype=torch.long, device=device)).unsqueeze(1)

    def final_model_fn(latent_z, points):
        return coarse(latent_z, points) + residual(latent_z, points)

    sdf, xs, ys, zs = evaluate_sdf_grid(final_model_fn, z, x_bounds, y_bounds, z_bounds, args.resolution, args.chunk_size, device)

    iso = args.iso_level
    sdf_min = float(sdf.min())
    sdf_max = float(sdf.max())
    if not (sdf_min <= iso <= sdf_max):
        iso = 0.5 * (sdf_min + sdf_max)
        print(f'iso_level {args.iso_level} not in SDF range [{sdf_min:.6f}, {sdf_max:.6f}]. Using fallback iso={iso:.6f}')

    verts_zyx, faces, normals, _ = measure.marching_cubes(sdf, level=iso, spacing=(zs[1]-zs[0], ys[1]-ys[0], xs[1]-xs[0]))
    verts_xyz = np.stack([verts_zyx[:,2] + xs[0], verts_zyx[:,1] + ys[0], verts_zyx[:,0] + zs[0]], axis=1)

    # trim boundary
    eps = 0.03
    mins = np.array([x_bounds[0], y_bounds[0], z_bounds[0]])
    maxs = np.array([x_bounds[1], y_bounds[1], z_bounds[1]])
    near_min = np.abs(verts_xyz - mins[None, :]) < eps
    near_max = np.abs(verts_xyz - maxs[None, :]) < eps
    near_boundary = np.any(near_min | near_max, axis=1)
    faces = np.array(faces)
    if faces.size > 0:
        face_keep = ~np.any(near_boundary[faces], axis=1)
        faces = faces[face_keep]

    os.makedirs(os.path.dirname(args.output_obj) or '.', exist_ok=True)
    save_obj(args.output_obj, verts_xyz, faces)
    print(f'Wrote mesh to: {args.output_obj}')

    if args.save_point_cloud:
        pc_path = os.path.splitext(args.output_obj)[0] + '.ply'
        save_point_cloud_ply(verts_xyz, normals if 'normals' in locals() else None, pc_path)
        print(f'Saved point cloud: {pc_path}')


if __name__ == '__main__':
    main()
