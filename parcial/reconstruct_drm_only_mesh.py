#!/usr/bin/env python3
"""Reconstruct a mesh from a DRM-only checkpoint trained on local SDF samples."""

import argparse
import os

import numpy as np
import torch

from drm_only_model import DRMSDFModel, LatentCodebook
from sdf_dataset import SDFDataset

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


def select_sample_index(sample_names, sample_index, sample_name):
    if sample_name:
        if sample_name not in sample_names:
            raise ValueError(f'Sample name {sample_name} not found in checkpoint sample list.')
        return sample_names.index(sample_name)
    if sample_index < 0 or sample_index >= len(sample_names):
        raise IndexError(f'--sample_index must be in [0, {len(sample_names) - 1}]')
    return sample_index


def sample_bounds_from_dataset(sample, padding=0.10):
    points = sample.get('sdf_points', None)
    if isinstance(points, torch.Tensor) and points.numel() > 0:
        pts = points.detach().cpu().numpy()
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
            f.write(f'f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n')


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
    parser.add_argument('--data_dir', type=str, default='', help='Folder containing the .npz SDF samples used for training')
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--sample_name', type=str, default='')
    parser.add_argument('--output_obj', type=str, default='mesh_results/parcial_reconstruction.obj')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--chunk_size', type=int, default=65536)
    parser.add_argument('--iso_level', type=float, default=0.0)
    parser.add_argument('--padding', type=float, default=0.10)
    parser.add_argument('--no_dataset_bounds', action='store_true', help='Use a fixed cube instead of sample bounds')
    parser.add_argument('--save_point_cloud', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    print(f'Using device: {device}')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    sample_names = checkpoint.get('sample_names', [])
    if not sample_names:
        raise RuntimeError('Checkpoint is missing sample_names. Train with parcial/train_drm_only.py.')

    latent_state = checkpoint.get('latent_state', {})
    embedding_weight = latent_state.get('embedding.weight', None)
    if embedding_weight is None:
        raise RuntimeError('Checkpoint is missing latent_state["embedding.weight"].')

    num_embeddings = int(embedding_weight.shape[0])
    if len(sample_names) != num_embeddings:
        print(
            f'Warning: checkpoint sample_names has {len(sample_names)} entries but '
            f'latent_state has {num_embeddings} embeddings. Using the latent embedding count.'
        )
        if len(sample_names) > num_embeddings:
            sample_names = sample_names[:num_embeddings]
        else:
            sample_names = sample_names + [f'sample_{i}' for i in range(len(sample_names), num_embeddings)]

    latent_dim = int(config.get('latent_dim', 256))
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

    latent = LatentCodebook(num_embeddings=num_embeddings, latent_dim=latent_dim).to(device)
    latent.load_state_dict(latent_state)
    latent.eval()

    idx = select_sample_index(sample_names, args.sample_index, args.sample_name)
    z = latent(torch.tensor([idx], dtype=torch.long, device=device)).unsqueeze(1)

    x_bounds, y_bounds, z_bounds = (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)
    if not args.no_dataset_bounds and args.data_dir:
        dataset = SDFDataset(args.data_dir)
        if idx >= len(dataset):
            raise IndexError(f'Sample index {idx} is out of dataset bounds ({len(dataset)} samples).')
        sample = dataset[idx]
        x_bounds, y_bounds, z_bounds = sample_bounds_from_dataset(sample, padding=args.padding)
    elif not args.no_dataset_bounds and not args.data_dir:
        print('No --data_dir provided; using default bounds [-1.2, 1.2]^3.')

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

    os.makedirs(os.path.dirname(args.output_obj) or '.', exist_ok=True)
    save_obj(args.output_obj, verts_xyz, faces)
    print(f'Wrote mesh to: {args.output_obj}')
    print(f'Sample: {sample_names[idx]} (index={idx})')
    print(f'Vertices={len(verts_xyz)} Faces={len(faces)}')

    if args.save_point_cloud:
        pc_path = os.path.splitext(args.output_obj)[0] + '.ply'
        save_point_cloud_ply(verts_xyz, normals, pc_path)
        print(f'Wrote point cloud to: {pc_path}')


if __name__ == '__main__':
    main()