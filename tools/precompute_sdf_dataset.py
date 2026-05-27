"""Precompute per-sample SDF query points and values for MPV3D.

This script reads the dataset split file from MPV3D/<split>.txt, loads the
front/back depth maps, reconstructs surface points using the orthographic
camera settings from MPV3D/camera.txt, samples query points near the surface,
computes signed distance values, and saves them as .npz files under:

    MPV3D/sdf/<split>/<im_name_stem>.npz

Each .npz contains:
    - points: (N, 3) float32
    - sdf:    (N,) float32

The training code then loads these files from the dataset.
"""

import argparse
import os
import sys

import numpy as np

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

        depth_f = np.load(depth_f_path) if os.path.exists(depth_f_path) else None
        depth_b = np.load(depth_b_path) if os.path.exists(depth_b_path) else None

        surface_pts = build_surface_points(depth_f, depth_b)
        points = sample_points_near_surface(surface_pts, num_points, sigma=sigma)
        sdf = compute_signed_sdf(points, surface_pts, depth_map=depth_f, cam_pose=FRONT_CAM_POSE, xmag=1.0, ymag=1.0)

        np.savez_compressed(out_path, points=points.astype(np.float32), sdf=sdf.astype(np.float32))
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