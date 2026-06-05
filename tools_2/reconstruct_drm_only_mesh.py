
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 tools_2/reconstruct_drm_only_mesh.py --checkpoint /data/113-1/users/asingh/project/3d/checkpoints/all_losses/aligned/DRM_only_bootstrap/best_net_DRMOnly.pth --dataroot /data/113-1/users/asingh/project/3d/MPV3D --datalist train_pairs --sample_index 0 --output_obj mesh_results/recon_0.obj --resolution 96 --chunk_size 65536 --gpu_id 0 --save_point_cloud


#!/usr/bin/env python3
"""Reconstruct a coarse mesh from a DRM-only checkpoint."""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append('.')

from data import create_dataset
from models_2 import DRMSDFModel, LatentCodebook

try:
    from skimage import measure
except ImportError as exc:
    raise ImportError(
        'scikit-image is required for marching cubes. Install with "pip install scikit-image".'
    ) from exc


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def build_dataset_opt(args):
    opt = SimpleNamespace()
    opt.dataroot = args.dataroot
    opt.datalist = args.datalist
    opt.datamode = args.datamode
    opt.model = args.dataset_model
    opt.batch_size = 1
    opt.img_width = 320
    opt.img_height = 512
    opt.isTrain = True
    opt.max_dataset_size = float('inf')
    opt.num_threads = 0
    opt.serial_batches = True
    opt.no_pin_memory = True
    opt.radius = 5
    opt.warproot = args.warproot
    return opt


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


def evaluate_sdf_grid(model, z, x_bounds, y_bounds, z_bounds, resolution, chunk_size, device):
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
            pred = model(z, p)
            sdf_values.append(pred.squeeze(0).squeeze(-1).detach().cpu().numpy())

    sdf = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)
    return sdf, xs, ys, zs


def save_obj(path, vertices_xyz, faces):
    with open(path, 'w') as f:
        for v in vertices_xyz:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for tri in faces:
            f.write(f'f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='mpv3d_example')
    parser.add_argument('--datalist', type=str, default='train_pairs')
    parser.add_argument('--datamode', type=str, default='aligned')
    parser.add_argument('--dataset_model', type=str, default='MTM', choices=['MTM', 'DRM'])
    parser.add_argument('--warproot', type=str, default='')
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--sample_name', type=str, default='')
    parser.add_argument('--output_obj', type=str, default='mesh_results/drm_only_reconstruction.obj')
    parser.add_argument('--resolution', type=int, default=96)
    parser.add_argument('--chunk_size', type=int, default=65536)
    parser.add_argument('--iso_level', type=float, default=0.0)
    parser.add_argument('--padding', type=float, default=0.10)
    parser.add_argument('--no_dataset_bounds', action='store_true', help='use fixed cube bounds instead of reading sample points from dataset')
    parser.add_argument('--save_point_cloud', action='store_true', default=False, help='also save reconstructed surface vertices as a PLY point cloud')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    device = get_device(args.gpu_id)
    print(f'Using device: {device}')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    sample_names = checkpoint.get('sample_names', [])
    if not sample_names:
        raise RuntimeError('Checkpoint is missing sample_names. Use a checkpoint produced by train_drm_only.py')

    latent_dim = int(config.get('latent_dim', 128))
    sdf_hidden_dim = int(config.get('sdf_hidden_dim', 512))
    sdf_num_layers = int(config.get('sdf_num_layers', 8))
    pe_L = int(config.get('pe_L', 6))

    model = DRMSDFModel(
        latent_dim=latent_dim,
        point_dim=3,
        hidden_dim=sdf_hidden_dim,
        num_layers=sdf_num_layers,
        pe_L=pe_L,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    latent = LatentCodebook(num_embeddings=len(sample_names), latent_dim=latent_dim).to(device)
    latent.load_state_dict(checkpoint['latent_state'])
    latent.eval()

    idx = select_sample_index(sample_names, args.sample_index, args.sample_name)
    z = latent(torch.tensor([idx], dtype=torch.long, device=device)).unsqueeze(1)

    x_bounds, y_bounds, z_bounds = (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)
    if not args.no_dataset_bounds:
        try:
            ds_opt = build_dataset_opt(args)
            dataset = create_dataset(ds_opt).dataset
            if idx >= len(dataset):
                raise IndexError(f'Sample index {idx} is out of dataset bounds ({len(dataset)} samples).')
            sample = dataset[idx]
            x_bounds, y_bounds, z_bounds = sample_bounds_from_dataset(sample, padding=args.padding)
        except Exception as exc:
            print(
                'Failed to infer bounds from dataset sample, using default cube [-1.2, 1.2]^3. '
                f'Reason: {exc}'
            )

    sdf, xs, ys, zs = evaluate_sdf_grid(
        model=model,
        z=z,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        resolution=args.resolution,
        chunk_size=args.chunk_size,
        device=device,
    )

    iso = args.iso_level
    sdf_min = float(sdf.min())
    sdf_max = float(sdf.max())
    if not (sdf_min <= iso <= sdf_max):
        iso = 0.5 * (sdf_min + sdf_max)
        print(
            f'iso_level {args.iso_level} not in SDF range [{sdf_min:.6f}, {sdf_max:.6f}]. '
            f'Using fallback iso={iso:.6f}'
        )

    verts_zyx, faces, normals, _ = measure.marching_cubes(
        sdf,
        level=iso,
        spacing=(zs[1] - zs[0], ys[1] - ys[0], xs[1] - xs[0]),
    )

    verts_xyz = np.stack(
        [
            verts_zyx[:, 2] + xs[0],
            verts_zyx[:, 1] + ys[0],
            verts_zyx[:, 0] + zs[0],
        ],
        axis=1,
    )

    # !! New added Code
    eps = 0.03

    mins = np.array([x_bounds[0], y_bounds[0], z_bounds[0]])
    maxs = np.array([x_bounds[1], y_bounds[1], z_bounds[1]])

    near_min = np.abs(verts_xyz - mins[None, :]) < eps
    near_max = np.abs(verts_xyz - maxs[None, :]) < eps
    near_boundary = np.any(near_min | near_max, axis=1)

    face_keep = ~np.any(near_boundary[faces], axis=1)
    faces = faces[face_keep]
    # !! Finish of new added code

    os.makedirs(os.path.dirname(args.output_obj) or '.', exist_ok=True)
    save_obj(args.output_obj, verts_xyz, faces)
    print(f'Wrote mesh to: {args.output_obj}')
    print(f'Sample: {sample_names[idx]} (index={idx})')
    print(f'Vertices={len(verts_xyz)} Faces={len(faces)}')

    if args.save_point_cloud:
        pc_path = os.path.splitext(args.output_obj)[0] + '.ply'
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

        save_point_cloud_ply(verts_xyz, normals if 'normals' in locals() else None, pc_path)
        print(f'Saved point cloud: {pc_path}')


if __name__ == '__main__':
    main()
