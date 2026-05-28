# python3 tools/precompute_sdf_dataset.py --dataroot /home/asingh/Desktop/uni/3d_vision/project/MPV3D --split train_pairs --num_points 512 --overwrite

"""Precompute per-sample SDF query points and values for MPV3D.

This script reads the dataset split file from MPV3D/<split>.txt, loads the
front/back depth maps, reconstructs surface points using the orthographic
camera settings from MPV3D/camera.txt, samples query points near the surface,
computes signed distance values, and saves them as .npz files under:

    MPV3D/sdf/<split>/<im_name_stem>.npz

Each .npz contains:
    - points: (N, 3) float32
    - sdf:    (N,) float32
    - surface_points: (M, 3) float32
    - surface_normals: (M, 3) float32

The training code then loads these files from the dataset.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.append('.')

from util.sdf_from_depth import sample_points_near_surface  # noqa: E402


X_OFFSET = 95.0
X_SCALE = 256.0
Y_SCALE = 256.0


def depth_to_world_grid(depth, is_back=False):
    """Match rgbd2pcd.py world conversion for MPV3D depth maps."""
    if depth is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    depth = depth.astype(np.float32).copy()
    if is_back:
        depth = np.flip(depth, axis=1)

    valid = depth > 0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    H, W = depth.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')

    # Use the same projection as rgbd2pcd.py
    x = (xx + X_OFFSET) / X_SCALE - 1.0
    y = (512.0 - 1.0 - yy) / Y_SCALE - 1.0
    z = 2.0 * depth - 1.0
    if not is_back:
        z = -1.0 * z

    points = np.stack([x, y, z], axis=-1)
    points = points[valid]
    return points.astype(np.float32), valid


def depth_to_normals_world(depth, is_back=False):
    """Approximate normals on the same world grid used by rgbd2pcd.py."""
    if depth is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

    depth = depth.astype(np.float32).copy()
    if is_back:
        depth = np.flip(depth, axis=1)

    valid = depth > 0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), valid

    z = 2.0 * depth - 1.0
    if not is_back:
        z = -1.0 * z

    # finite differences on the aligned world z grid
    dz_dy, dz_dx = np.gradient(z)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(z, dtype=np.float32)
    normals = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    normals = normals[valid]
    return normals.astype(np.float32), valid


def load_split_entries(dataroot, split_name):
    split_path = os.path.join(dataroot, f'{split_name}.txt')
    entries = []
    with open(split_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            im_name, c_name = line.split()
            entries.append((im_name, c_name))
    return entries


def build_surface_points(depth_f, depth_b):
    """Build the combined surface point cloud using the same fusion logic as rgbd2pcd.py."""
    if depth_f is None and depth_b is None:
        return np.zeros((0, 3), dtype=np.float32)

    fd = depth_f.astype(np.float32).copy() if depth_f is not None else None
    bd = depth_b.astype(np.float32).copy() if depth_b is not None else None

    if fd is None:
        fd = np.zeros_like(bd)
    if bd is None:
        bd = np.zeros_like(fd)

    # match rgbd2pcd.py: slightly shrink front depth, flip back depth, and drop invalid overlap
    fd = fd - 0.02
    bd = np.flip(bd, axis=1)
    rm_idx = (fd - bd) < 0
    fd[rm_idx] = 0
    bd[rm_idx] = 0

    surface_f, _ = depth_to_world_grid(fd, is_back=False)
    surface_b, _ = depth_to_world_grid(bd, is_back=True)

    if surface_f.size and surface_b.size:
        return np.concatenate([surface_f, surface_b], axis=0)
    if surface_f.size:
        return surface_f
    return surface_b


def sample_valid_surface_points(depth, max_points, is_back=False):
    if depth is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    surface_pts, valid_mask = depth_to_world_grid(depth, is_back=is_back)
    normals_world, _ = depth_to_normals_world(depth, is_back=is_back)

    if surface_pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    if surface_pts.shape[0] > max_points:
        idx = np.random.choice(surface_pts.shape[0], size=max_points, replace=False)
        surface_pts = surface_pts[idx]
        normals_world = normals_world[idx]

    return surface_pts.astype(np.float32), normals_world.astype(np.float32)


def sample_sdf_queries(surface_points, surface_normals, num_points, sigma=0.01):
    """Sample local SDF queries around the surface with guaranteed signed labels.

    We generate three groups:
    - surface points with sdf = 0
    - positive samples by moving along the normal direction
    - negative samples by moving opposite to the normal direction
    """
    if surface_points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    surface_points = surface_points.astype(np.float32)
    surface_normals = surface_normals.astype(np.float32)
    normal_norm = np.linalg.norm(surface_normals, axis=-1, keepdims=True)
    surface_normals = surface_normals / np.maximum(normal_norm, 1e-8)

    n_surface = max(1, num_points // 5)
    n_pos = max(1, (num_points - n_surface) // 2)
    n_neg = num_points - n_surface - n_pos
    if n_neg <= 0:
        n_neg = 1
        if n_pos > 1:
            n_pos -= 1

    idx_surface = np.random.randint(0, surface_points.shape[0], size=n_surface)
    idx_pos = np.random.randint(0, surface_points.shape[0], size=n_pos)
    idx_neg = np.random.randint(0, surface_points.shape[0], size=n_neg)

    # Small tangential noise helps avoid all samples lying exactly on the normal line.
    tangential_noise = np.random.normal(scale=sigma * 0.25, size=surface_points.shape).astype(np.float32)

    surface_q = surface_points[idx_surface] + tangential_noise[idx_surface]
    surface_sdf = np.zeros((n_surface,), dtype=np.float32)

    pos_t = np.random.uniform(0.25 * sigma, sigma, size=(n_pos, 1)).astype(np.float32)
    pos_q = surface_points[idx_pos] + surface_normals[idx_pos] * pos_t + tangential_noise[idx_pos]
    pos_sdf = pos_t.squeeze(1)

    neg_t = np.random.uniform(0.25 * sigma, sigma, size=(n_neg, 1)).astype(np.float32)
    neg_q = surface_points[idx_neg] - surface_normals[idx_neg] * neg_t + tangential_noise[idx_neg]
    neg_sdf = -neg_t.squeeze(1)

    points = np.concatenate([surface_q, pos_q, neg_q], axis=0)
    sdf = np.concatenate([surface_sdf, pos_sdf, neg_sdf], axis=0)
    return points.astype(np.float32), sdf.astype(np.float32)


def precompute_split(dataroot, split_name, num_points, sigma, overwrite=False):
    entries = load_split_entries(dataroot, split_name)
    out_root = os.path.join(dataroot, 'sdf', split_name)
    os.makedirs(out_root, exist_ok=True)

    for idx, (im_name, _c_name) in enumerate(entries):
        out_name = im_name.replace('.png', '.npz')
        out_path = os.path.join(out_root, out_name)
        if os.path.exists(out_path) and not overwrite:
            print(f'[{idx+1}/{len(entries)}] skip {out_name} (exists)')
            continue

        depth_f_path = os.path.join(dataroot, 'depth', im_name.replace('.png', '_depth.npy'))
        depth_b_path = os.path.join(dataroot, 'depth', im_name.replace('front.png', 'back_depth.npy'))

        depth_f = np.load(depth_f_path).astype(np.float32) if os.path.exists(depth_f_path) else None
        depth_b = np.load(depth_b_path).astype(np.float32) if os.path.exists(depth_b_path) else None

        surface_points_f, surface_normals_f = sample_valid_surface_points(
            depth_f,
            num_points,
            is_back=False,
        )
        surface_points_b, surface_normals_b = sample_valid_surface_points(
            depth_b,
            num_points,
            is_back=True,
        )

        if surface_points_f.size and surface_points_b.size:
            surface_points = np.concatenate([surface_points_f, surface_points_b], axis=0)
            surface_normals = np.concatenate([surface_normals_f, surface_normals_b], axis=0)
        elif surface_points_f.size:
            surface_points = surface_points_f
            surface_normals = surface_normals_f
        else:
            surface_points = surface_points_b
            surface_normals = surface_normals_b

        points, sdf = sample_sdf_queries(surface_points, surface_normals, num_points, sigma=sigma)
        # normalize sdf to [-1, 1] by max absolute value and save the scale
        if sdf.size:
            max_abs = float(max(abs(sdf.min()), abs(sdf.max())))
            sdf_scale = max_abs if max_abs > 0 else 1.0
            sdf_norm = (sdf.astype(np.float32) / sdf_scale).astype(np.float32)
        else:
            sdf_scale = 1.0
            sdf_norm = sdf.astype(np.float32)

        np.savez_compressed(
            out_path,
            points=points.astype(np.float32),
            sdf=sdf_norm.astype(np.float32),
            surface_points=surface_points.astype(np.float32),
            surface_normals=surface_normals.astype(np.float32),
            sdf_scale=np.array([sdf_scale], dtype=np.float32),
        )
        print(f'[{idx+1}/{len(entries)}] wrote {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/home/asingh/Desktop/uni/3d_vision/project/MPV3D')
    parser.add_argument('--split', type=str, default='train_pairs', help='split txt file without extension')
    parser.add_argument('--num_points', type=int, default=256)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    precompute_split(args.dataroot, args.split, args.num_points, args.sigma, overwrite=args.overwrite)


if __name__ == '__main__':
    main()