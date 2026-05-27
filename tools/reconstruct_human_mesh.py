
# python3 tools/reconstruct_human_mesh.py --dataroot /home/asingh/Desktop/uni/3d_vision/project/MPV3D --datalist test_pairs --sample_index 0 --num_images 1 --mtm_ckpt /home/asingh/Desktop/uni/3d_vision/project/latest_net_MTM.pth --drm_ckpt ./checkpoints/aligned/DRM_train/best_net_DRM.pth --output_dir ./mesh_results

"""Reconstruct a 3D human mesh from trained MTM + DRM checkpoints.

This script:
1. Loads one MPV3D sample from the aligned dataset.
2. Runs the trained MTM to obtain latent `z`.
3. Evaluates the trained DRM SDF network on a dense 3D grid.
4. Extracts the zero level set with marching cubes.
5. Writes a mesh as Wavefront OBJ.

Example:
    python3 tools/reconstruct_human_mesh.py \
        --dataroot /home/asingh/Desktop/uni/3d_vision/project/MPV3D \
        --datalist test_pairs \
        --sample_index 0 \
        --mtm_ckpt /home/asingh/Desktop/uni/3d_vision/project/latest_net_MTM.pth \
        --drm_ckpt ./checkpoints/aligned/DRM_train/best_net_DRM.pth \
        --output_dir ./mesh_results

Dependencies:
    - scikit-image (for marching cubes)
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append('.')

from data import create_dataset
from models import networks

try:
    from skimage import measure
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        'scikit-image is required for marching cubes. Install it with '\
        '"pip install scikit-image" and rerun the script.'
    ) from exc


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


def build_dataset_opt(args):
    ds_opt = SimpleNamespace()
    ds_opt.dataroot = args.dataroot
    ds_opt.datalist = args.datalist
    ds_opt.datamode = 'aligned'
    ds_opt.model = 'MTM'
    ds_opt.batch_size = 1
    ds_opt.img_width = 320
    ds_opt.img_height = 512
    ds_opt.isTrain = True
    ds_opt.max_dataset_size = float('inf')
    ds_opt.num_threads = 0
    ds_opt.serial_batches = True
    ds_opt.no_pin_memory = True
    ds_opt.radius = 5
    ds_opt.warproot = ''
    return ds_opt


def build_model_opt(args, gpu_ids):
    opt = SimpleNamespace()
    opt.latent_dim = args.latent_dim
    opt.point_dim = 3
    opt.sdf_hidden_dim = args.sdf_hidden_dim
    opt.sdf_num_layers = args.sdf_num_layers
    opt.norm = 'instance'
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    opt.gpu_ids = gpu_ids
    opt.isTrain = False
    opt.lr = 0.001
    opt.checkpoints_dir = './checkpoints'
    opt.datamode = 'aligned'
    opt.name = 'DRM_train'
    opt.display_ncols = 2
    opt.ngf = 64
    return opt


def select_sample(dataset, sample_index):
    sample = dataset[sample_index]
    if isinstance(sample, dict):
        return sample
    raise TypeError('Dataset sample is not a dict; cannot reconstruct mesh.')


def load_state_dict(path, device):
    state = torch.load(path, map_location=device)
    if hasattr(state, '_metadata'):
        del state._metadata
    return state


def load_mtm(opt, ckpt_path, device):
    net = networks.define_MTM(
        input_nc_A=29,
        input_nc_B=3,
        ngf=opt.ngf,
        n_layers=3,
        img_height=512,
        img_width=320,
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
    state = load_state_dict(ckpt_path, device)
    load_res = net.load_state_dict(state, strict=False)
    if hasattr(load_res, 'missing_keys'):
        print('MTM missing keys:', load_res.missing_keys)
    if hasattr(load_res, 'unexpected_keys'):
        print('MTM unexpected keys:', load_res.unexpected_keys)
    net.to(device)
    net.eval()
    return net


def load_drm(opt, ckpt_path, device):
    net = networks.define_DRM(
        latent_dim=opt.latent_dim,
        point_dim=opt.point_dim,
        hidden_dim=opt.sdf_hidden_dim,
        num_layers=opt.sdf_num_layers,
        output_dim=1,
        norm='instance',
        init_type='normal',
        init_gain=0.02,
        gpu_ids=opt.gpu_ids,
    )
    state = load_state_dict(ckpt_path, device)
    load_res = net.load_state_dict(state, strict=False)
    if hasattr(load_res, 'missing_keys'):
        print('DRM missing keys:', load_res.missing_keys)
    if hasattr(load_res, 'unexpected_keys'):
        print('DRM unexpected keys:', load_res.unexpected_keys)
    net.to(device)
    net.eval()
    return net


def sample_bounds(sample, padding=0.1):
    z_values = []
    for key in ('person_fdepth', 'person_bdepth'):
        value = sample.get(key, None)
        if isinstance(value, torch.Tensor):
            arr = value.squeeze().detach().cpu().numpy()
            arr = arr[np.isfinite(arr)]
            arr = arr[arr != 0]
            if arr.size > 0:
                z_values.append(arr)

    if z_values:
        z_all = np.concatenate(z_values)
        z_min = float(z_all.min() - padding)
        z_max = float(z_all.max() + padding)
    else:
        z_min, z_max = -1.5, 1.5

    return (-1.0, 1.0), (-1.0, 1.0), (z_min, z_max)


def evaluate_sdf_on_grid(net, latent_z, x_bounds, y_bounds, z_bounds, resolution, chunk_size, device):
    xs = np.linspace(x_bounds[0], x_bounds[1], resolution, dtype=np.float32)
    ys = np.linspace(y_bounds[0], y_bounds[1], resolution, dtype=np.float32)
    zs = np.linspace(z_bounds[0], z_bounds[1], resolution, dtype=np.float32)

    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing='ij')
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    sdf_values = []
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk_size):
            end = min(start + chunk_size, points.shape[0])
            points_chunk = torch.from_numpy(points[start:end]).float().to(device).unsqueeze(0)
            sdf_chunk = net(latent_z, points_chunk)
            sdf_values.append(sdf_chunk.squeeze(0).squeeze(-1).detach().cpu().numpy())

    sdf = np.concatenate(sdf_values, axis=0)
    sdf = sdf.reshape(resolution, resolution, resolution)
    return sdf, (xs, ys, zs)


def marching_cubes_to_obj(sdf, xs, ys, zs, level, output_path):
    verts, faces, normals, _ = measure.marching_cubes(sdf, level=level, spacing=(zs[1] - zs[0], ys[1] - ys[0], xs[1] - xs[0]))

    # marching_cubes returns coordinates in (z, y, x) order; convert to (x, y, z)
    verts_xyz = np.stack([
        verts[:, 2] + xs[0],
        verts[:, 1] + ys[0],
        verts[:, 0] + zs[0],
    ], axis=1)

    with open(output_path, 'w') as f:
        for v in verts_xyz:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            # OBJ is 1-indexed
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')

    return verts_xyz, faces, normals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/home/asingh/Desktop/uni/3d_vision/project/MPV3D')
    parser.add_argument('--datalist', type=str, default='test_pairs')
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=1, help='number of dataset samples to reconstruct')
    parser.add_argument('--start_index', type=int, default=0, help='dataset index to start from')
    parser.add_argument('--mtm_ckpt', type=str, default='/home/asingh/Desktop/uni/3d_vision/project/latest_net_MTM.pth')
    parser.add_argument('--drm_ckpt', type=str, default='./checkpoints/aligned/DRM_train/best_net_DRM.pth')
    parser.add_argument('--output_dir', type=str, default='./mesh_results')
    parser.add_argument('--output_name', type=str, default='reconstruction.obj')
    parser.add_argument('--resolution', type=int, default=96)
    parser.add_argument('--chunk_size', type=int, default=65536)
    parser.add_argument('--iso_level', type=float, default=0.0)
    parser.add_argument('--auto_iso', action='store_true', default=True, help='fallback to a valid iso level if requested level is out of range')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--sdf_hidden_dim', type=int, default=128)
    parser.add_argument('--sdf_num_layers', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Using device: {device} ({torch.cuda.get_device_name(0)})')
    else:
        print(f'Using device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    ds_opt = build_dataset_opt(args)
    dataset_loader = create_dataset(ds_opt)
    dataset = dataset_loader.dataset

    sample_indices = [args.sample_index]
    if args.num_images > 1:
        sample_indices = list(range(args.start_index, min(args.start_index + args.num_images, len(dataset))))

    mtm_opt = build_model_opt(args, [0] if device.type == 'cuda' else [])
    drm_opt = build_model_opt(args, [0] if device.type == 'cuda' else [])

    raw_mtm = load_mtm(mtm_opt, args.mtm_ckpt, device)
    drm_net = load_drm(drm_opt, args.drm_ckpt, device)

    for local_idx, sample_index in enumerate(sample_indices):
        sample = select_sample(dataset, sample_index)

        agnostic = sample['agnostic']
        cloth = sample['cloth']
        if not isinstance(agnostic, torch.Tensor) or not isinstance(cloth, torch.Tensor):
            raise TypeError('Sample does not contain MTM inputs as tensors.')

        agnostic = agnostic.unsqueeze(0).to(device)
        cloth = cloth.unsqueeze(0).to(device)

        with torch.no_grad():
            mtm_out = raw_mtm(agnostic, cloth)
            latent_z = mtm_out['z'].to(device)

        x_bounds, y_bounds, z_bounds = sample_bounds(sample)
        sdf, (xs, ys, zs) = evaluate_sdf_on_grid(
            drm_net,
            latent_z,
            x_bounds,
            y_bounds,
            z_bounds,
            resolution=args.resolution,
            chunk_size=args.chunk_size,
            device=device,
        )

        sdf_min = float(sdf.min())
        sdf_max = float(sdf.max())
        print(f'Sample {sample_index}: SDF range min={sdf_min:.6f}, max={sdf_max:.6f}')
        iso_level = args.iso_level
        if not (sdf_min <= iso_level <= sdf_max):
            if args.auto_iso:
                iso_level = 0.5 * (sdf_min + sdf_max)
                print(
                    f'Warning: requested iso level {args.iso_level} is outside the SDF range. '
                    f'Using fallback iso level {iso_level:.6f}.'
                )
            else:
                raise RuntimeError(
                    f'Iso level {args.iso_level} is outside the SDF range [{sdf_min:.6f}, {sdf_max:.6f}]. '
                    'Try increasing the grid bounds or changing --iso_level.'
                )

        output_name = args.output_name
        if args.num_images > 1:
            base, ext = os.path.splitext(args.output_name)
            output_name = f'{base}_{sample_index:04d}{ext or ".obj"}'
        output_path = os.path.join(args.output_dir, output_name)
        verts, faces, normals = marching_cubes_to_obj(sdf, xs, ys, zs, iso_level, output_path)
        print(f'Saved mesh: {output_path}')
        print(f'Vertices: {len(verts)} | Faces: {len(faces)} | Normals: {len(normals)}')


if __name__ == '__main__':
    main()