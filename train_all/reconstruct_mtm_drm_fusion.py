#!/usr/bin/env python3

"""Reconstruct a single human mesh from the image-conditioned MTM+DRM fusion checkpoint."""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append('.')

from data import create_dataset
from models import networks
from train_all.fusion_sdf_model import ImageConditionedFusionSDF

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


def build_dataset_opt(args):
    opt = SimpleNamespace()
    opt.dataroot = args.dataroot
    opt.datalist = args.datalist
    opt.datamode = args.datamode
    opt.model = 'MTM'
    opt.batch_size = 1
    opt.img_width = args.img_width
    opt.img_height = args.img_height
    opt.isTrain = False
    opt.max_dataset_size = float('inf')
    opt.num_threads = 0
    opt.serial_batches = True
    opt.no_pin_memory = True
    opt.radius = args.radius
    opt.warproot = ''
    return opt


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


def evaluate_sdf_grid(model_fn, z, image_tensor, x_bounds, y_bounds, z_bounds, resolution, chunk_size, device):
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
            pred = model_fn(z, image_tensor, p)
            if isinstance(pred, dict):
                pred = pred['final_sdf']
            sdf_values.append(pred.squeeze(0).squeeze(-1).detach().cpu().numpy())

    sdf = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)
    return sdf, xs, ys, zs


def save_obj(path, vertices_xyz, faces):
    with open(path, 'w') as f:
        for v in vertices_xyz:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for tri in faces:
            f.write(f'f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n')


def keep_largest_connected_component(faces):
    if faces is None or len(faces) == 0:
        return faces
    faces = np.asarray(faces, dtype=np.int64)
    v2f = {}
    for fi, tri in enumerate(faces):
        for v in tri:
            v2f.setdefault(int(v), []).append(fi)
    visited = np.zeros(len(faces), dtype=bool)
    comps = []
    for s in range(len(faces)):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        comp = []
        while stack:
            fi = stack.pop()
            comp.append(fi)
            for v in faces[fi]:
                for nfi in v2f.get(int(v), []):
                    if not visited[nfi]:
                        visited[nfi] = True
                        stack.append(nfi)
        comps.append(comp)
    largest = max(comps, key=len)
    return faces[np.array(largest, dtype=np.int64)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mtm_ckpt', type=str, default='', help='optional external MTM checkpoint if mtm_state is not saved in the fusion checkpoint')
    parser.add_argument('--drm_mode', type=str, default='residual', choices=['coarse', 'residual'])
    parser.add_argument('--dataroot', type=str, default='mpv3d_example')
    parser.add_argument('--datalist', type=str, default='test_pairs')
    parser.add_argument('--datamode', type=str, default='aligned')
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--sample_name', type=str, default='')
    parser.add_argument('--output_obj', type=str, default='mesh_results/recon_fusion.obj')
    parser.add_argument('--resolution', type=int, default=96)
    parser.add_argument('--chunk_size', type=int, default=65536)
    parser.add_argument('--iso_level', type=float, default=0.0)
    parser.add_argument('--padding', type=float, default=0.10)
    parser.add_argument('--no_dataset_bounds', action='store_true')
    parser.add_argument('--save_point_cloud', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--img_width', type=int, default=320)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--mtm_z_dim', type=int, default=1024)
    parser.add_argument('--sdf_hidden_dim', type=int, default=512)
    parser.add_argument('--sdf_num_layers', type=int, default=8)
    parser.add_argument('--pe_L', type=int, default=6)
    parser.add_argument('--image_in_channels', type=int, default=3)
    parser.add_argument('--image_feature_dim', type=int, default=256)
    parser.add_argument('--image_scale', type=float, default=0.1)
    parser.add_argument('--residual_scale', type=float, default=1.0)
    parser.add_argument('--fusion_scale', type=float, default=1.0)
    parser.add_argument('--debug_forward', action='store_true', default=False, help='print latent/fused/SDF stats during reconstruction')
    return parser.parse_args()


def load_state_dict(path):
    state = torch.load(path, map_location='cpu')
    if hasattr(state, '_metadata'):
        del state._metadata
    return state


def infer_drm_mode(checkpoint, default_mode='residual'):
    if isinstance(checkpoint, dict):
        drm_mode = checkpoint.get('drm_mode', None)
        if drm_mode in ('coarse', 'residual'):
            return drm_mode
        if 'residual_state' in checkpoint:
            return 'residual'
        return 'coarse'
    return default_mode


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    print(f'Using device: {device}')

    checkpoint = load_state_dict(args.checkpoint)
    sample_names = checkpoint.get('sample_names', []) if isinstance(checkpoint, dict) else []
    if sample_names is None:
        sample_names = []

    drm_mode = infer_drm_mode(checkpoint, default_mode=args.drm_mode)
    print(f'DRM mode: {drm_mode}')
    if drm_mode == 'residual':
        print('Residual branch enabled')
    else:
        print('Residual branch disabled')

    mtm_state = None
    if isinstance(checkpoint, dict):
        mtm_state = checkpoint.get('mtm_state', None)
    if mtm_state is None and args.mtm_ckpt:
        mtm_state = load_state_dict(args.mtm_ckpt)
        mtm_state = mtm_state.get('mtm_state') or mtm_state.get('model_state') or mtm_state
    if mtm_state is None:
        raise RuntimeError('No MTM state found. Use a fusion checkpoint with mtm_state or pass --mtm_ckpt.')

    mtm = networks.define_MTM(
        input_nc_A=29,
        input_nc_B=3,
        ngf=64,
        n_layers=3,
        img_height=args.img_height,
        img_width=args.img_width,
        grid_size=3,
        add_tps=True,
        add_depth=True,
        add_segmt=True,
        latent_dim=args.mtm_z_dim,
        norm='instance',
        use_dropout=False,
        init_type='normal',
        init_gain=0.02,
        gpu_ids=[args.gpu_id] if device.type == 'cuda' else [],
    )
    if hasattr(mtm_state, '_metadata'):
        del mtm_state._metadata
    mtm.load_state_dict(mtm_state, strict=False)
    mtm.to(device)
    mtm.eval()

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
        drm_mode=drm_mode,
        debug=args.debug_forward,
    ).to(device)
    fusion_state = checkpoint.get('image_fusion_state') or checkpoint.get('fusion_state') or checkpoint.get('model_state') or checkpoint
    if hasattr(fusion_state, '_metadata'):
        del fusion_state._metadata
    fusion.load_state_dict(fusion_state, strict=False)
    fusion.eval()

    dataset = None
    if not args.no_dataset_bounds:
        try:
            ds_opt = build_dataset_opt(args)
            dataset = create_dataset(ds_opt).dataset
        except Exception as exc:
            print('Failed to load dataset for bounds; using default cube. Reason:', exc)

    if dataset is not None and len(dataset) > 0:
        idx = select_sample_index(sample_names or [str(i) for i in range(len(dataset))], args.sample_index, args.sample_name)
        sample = dataset[idx]
        x_bounds, y_bounds, z_bounds = sample_bounds_from_dataset(sample, padding=args.padding)
    else:
        x_bounds, y_bounds, z_bounds = (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)

    if len(sample_names) == 0:
        sample_names = [str(i) for i in range(max(1, getattr(fusion.coarse, 'num_embeddings', 1)))]

    idx = select_sample_index(sample_names, args.sample_index, args.sample_name)
    if dataset is not None and len(dataset) > idx:
        sample = dataset[idx]
    else:
        raise RuntimeError('Could not load sample for reconstruction.')

    agnostic = sample['agnostic'].unsqueeze(0).to(device)
    person = sample['person'].unsqueeze(0).to(device)
    cloth = sample['cloth'].unsqueeze(0).to(device)
    image_tensor = person

    with torch.no_grad():
        mtm_out = mtm(agnostic, cloth)
        latent_z = mtm_out['z']
        if latent_z.dim() == 3 and latent_z.size(1) == 1:
            latent_z = latent_z.squeeze(1)

    sdf, xs, ys, zs = evaluate_sdf_grid(
        fusion,
        latent_z,
        image_tensor,
        x_bounds,
        y_bounds,
        z_bounds,
        args.resolution,
        args.chunk_size,
        device,
    )

    # !! For reconstruction
    print("only for reconstruction")
    print("=" * 50)
    print("SDF statistics")
    print("=" * 50)
    print("min:", sdf.min())
    print("max:", sdf.max())
    print("mean:", sdf.mean())
    print("negative:", (sdf < 0).sum())
    print("positive:", (sdf >= 0).sum())

    sdf_min = float(sdf.min())
    sdf_max = float(sdf.max())
    iso = args.iso_level
    if not (sdf_min <= iso <= sdf_max):
        iso = 0.5 * (sdf_min + sdf_max)

    verts_zyx, faces, normals, _ = measure.marching_cubes(
        sdf,
        level=iso,
        spacing=(zs[1] - zs[0], ys[1] - ys[0], xs[1] - xs[0]),
    )
    verts_xyz = np.stack([verts_zyx[:, 2] + xs[0], verts_zyx[:, 1] + ys[0], verts_zyx[:, 0] + zs[0]], axis=1)
    faces = keep_largest_connected_component(faces)

    os.makedirs(os.path.dirname(args.output_obj) or '.', exist_ok=True)
    save_obj(args.output_obj, verts_xyz, faces)
    print(f'Wrote mesh to: {args.output_obj}')

    if args.save_point_cloud:
        pc_path = os.path.splitext(args.output_obj)[0] + '.ply'
        with open(pc_path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(verts_xyz)}\n')
            f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
            for p in verts_xyz:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        print(f'Saved point cloud: {pc_path}')


if __name__ == '__main__':
    main()
