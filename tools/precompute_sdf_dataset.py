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

from util.sdf_from_depth import (  # noqa: E402
    backproject_ortho_depth,
    compute_signed_sdf,
    sample_points_near_surface,
)


FRONT_CAM_POSE = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 2.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)
BACK_CAM_POSE = np.array(
    [[-1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, -2.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)


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


def build_surface_points(depth_f, depth_b, xmag=1.0, ymag=1.0):
    if depth_f is not None:
        surface_f, _ = backproject_ortho_depth(depth_f, xmag=xmag, ymag=ymag, cam_pose=FRONT_CAM_POSE)
    else:
        surface_f = np.zeros((0, 3), dtype=np.float32)

    if depth_b is not None:
        surface_b, _ = backproject_ortho_depth(depth_b, xmag=xmag, ymag=ymag, cam_pose=BACK_CAM_POSE)
    else:
        surface_b = np.zeros((0, 3), dtype=np.float32)

    if surface_f.size and surface_b.size:
        return np.concatenate([surface_f, surface_b], axis=0)
    if surface_f.size:
        return surface_f
    return surface_b


def compute_surface_normals(depth, xmag=1.0, ymag=1.0):
    """Approximate camera-space normals from an orthographic depth map."""
    if depth is None:
        return np.zeros((0, 3), dtype=np.float32)

    depth = depth.astype(np.float32)
    H, W = depth.shape
    dx = (2.0 * xmag) / max(W - 1, 1)
    dy = (2.0 * ymag) / max(H - 1, 1)

    # gradients are computed in image order: d/dy, d/dx
    dz_dy, dz_dx = np.gradient(depth, dy, dx)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(depth, dtype=np.float32)
    normals = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    return normals.astype(np.float32)


def world_normals_from_camera(normals_cam, cam_pose):
    rot = cam_pose[:3, :3].astype(np.float32)
    normals_world = normals_cam.reshape(-1, 3) @ rot.T
    norm = np.linalg.norm(normals_world, axis=-1, keepdims=True)
    normals_world = normals_world / np.maximum(norm, 1e-8)
    return normals_world.astype(np.float32)


def sample_valid_surface_points(depth, normals, cam_pose, max_points):
    if depth is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    surface_pts, _ = backproject_ortho_depth(depth, xmag=1.0, ymag=1.0, cam_pose=cam_pose)
    valid_mask = depth.reshape(-1) > 0
    normals_world = world_normals_from_camera(normals, cam_pose)

    normals_world = normals_world[valid_mask.reshape(-1)]

    # `surface_pts` is already filtered by backproject_ortho_depth using the same valid mask.
    # Keep it as-is and only filter normals to the corresponding valid pixels.

    if surface_pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    if surface_pts.shape[0] > max_points:
        idx = np.random.choice(surface_pts.shape[0], size=max_points, replace=False)
        surface_pts = surface_pts[idx]
        normals_world = normals_world[idx]

    return surface_pts.astype(np.float32), normals_world.astype(np.float32)


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
        if depth_b is not None:
            depth_b = np.flip(depth_b, axis=1)  # match dataset loader alignment

        surface_pts = build_surface_points(depth_f, depth_b)
        points = sample_points_near_surface(surface_pts, num_points, sigma=sigma)
        sdf = compute_signed_sdf(points, surface_pts, depth_map=depth_f, cam_pose=FRONT_CAM_POSE, xmag=1.0, ymag=1.0)

        surface_points_f, surface_normals_f = sample_valid_surface_points(
            depth_f,
            compute_surface_normals(depth_f) if depth_f is not None else np.zeros((0, 3), dtype=np.float32),
            FRONT_CAM_POSE,
            num_points,
        )
        surface_points_b, surface_normals_b = sample_valid_surface_points(
            depth_b,
            compute_surface_normals(depth_b) if depth_b is not None else np.zeros((0, 3), dtype=np.float32),
            BACK_CAM_POSE,
            num_points,
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

        np.savez_compressed(
            out_path,
            points=points.astype(np.float32),
            sdf=sdf.astype(np.float32),
            surface_points=surface_points.astype(np.float32),
            surface_normals=surface_normals.astype(np.float32),
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