
# Reconstruction
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 tools_2_image_encoder/reconstruct_occupancy_from_image.py --checkpoint /data/125-1/users/asingh/proves/drm_IE/occupancy_function/aligned/DRM_image_occupancy/best_net_DRMOccupancy.pth --image_path /data/113-1/users/asingh/project/3d/MPV3D/image/LA221D021-M12@12=person_whole_front.png --bounds_npz /data/113-1/users/asingh/project/3d/MPV3D/sdf/train_pairs/0VB21E007-T11@9=person_whole_front.npz --output_obj mesh_results/unseen_person_occ.obj --resolution 96



#!/usr/bin/env python3
"""Reconstruct one mesh from an image-conditioned occupancy checkpoint."""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from common import bounds_from_npz, get_device, load_conditioning_image  # noqa: E402
from image_encoder_model import build_image_encoder_from_args  # noqa: E402
from models_2 import DRMSDFModel  # noqa: E402
from tools_2.reconstruct_drm_only_mesh import save_obj  # noqa: E402

try:
    from skimage import measure
except ImportError as exc:
    raise ImportError(
        'scikit-image is required for marching cubes. Install with "pip install scikit-image".'
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_obj", type=str, default="mesh_results/occupancy_reconstruction.obj")
    parser.add_argument("--output_point_cloud", type=str, default="")
    parser.add_argument("--no_point_cloud", action="store_true")

    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--iso_logit", type=float, default=0.0)
    parser.add_argument("--use_probability_grid", action="store_true")
    parser.add_argument("--iso_probability", type=float, default=0.5)
    parser.add_argument("--allow_iso_fallback", action="store_true")

    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--bounds_npz", type=str, default="")
    parser.add_argument(
        "--bounds_preset",
        type=str,
        default="human",
        choices=["human", "human_tight", "cube"],
        help="Default reconstruction bounds when --bounds_npz or manual bounds are not provided.",
    )
    parser.add_argument("--x_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--y_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--z_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--boundary_eps", type=float, default=0.03)
    parser.add_argument("--keep_largest_component", dest="keep_largest_component", action="store_true", default=True)
    parser.add_argument("--no_keep_largest_component", dest="keep_largest_component", action="store_false")

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--sdf_hidden_dim", type=int, default=None)
    parser.add_argument("--sdf_num_layers", type=int, default=None)
    parser.add_argument("--pe_L", type=int, default=None)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--image_channels", type=int, default=None)
    parser.add_argument("--encoder_base_channels", type=int, default=None)
    parser.add_argument("--encoder_num_blocks", type=int, default=None)
    parser.add_argument("--encoder_head_hidden_dim", type=int, default=None)
    parser.add_argument("--encoder_dropout", type=float, default=None)
    parser.add_argument("--encoder_use_batchnorm", type=int, default=None, choices=[0, 1])
    return parser.parse_args()


def config_value(args, config, key, default):
    value = getattr(args, key)
    if value is not None:
        return value
    return config.get(key, default)


def build_runtime_args(args, config):
    return SimpleNamespace(
        latent_dim=int(config_value(args, config, "latent_dim", 128)),
        sdf_hidden_dim=int(config_value(args, config, "sdf_hidden_dim", 512)),
        sdf_num_layers=int(config_value(args, config, "sdf_num_layers", 8)),
        pe_L=int(config_value(args, config, "pe_L", 6)),
        img_width=int(config_value(args, config, "img_width", 320)),
        img_height=int(config_value(args, config, "img_height", 512)),
        image_channels=int(config_value(args, config, "image_channels", 3)),
        encoder_base_channels=int(config_value(args, config, "encoder_base_channels", 32)),
        encoder_num_blocks=int(config_value(args, config, "encoder_num_blocks", 5)),
        encoder_head_hidden_dim=int(config_value(args, config, "encoder_head_hidden_dim", 512)),
        encoder_dropout=float(config_value(args, config, "encoder_dropout", 0.0)),
        encoder_use_batchnorm=int(config_value(args, config, "encoder_use_batchnorm", 1)),
    )


def preset_bounds(name):
    if name == "cube":
        return (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)
    if name == "human_tight":
        return (-0.65, 0.65), (-1.05, 1.05), (-0.65, 0.65)
    return (-0.8, 0.8), (-1.15, 1.15), (-0.8, 0.8)


def resolve_bounds(args):
    if args.bounds_npz:
        return bounds_from_npz(args.bounds_npz, padding=args.padding)

    preset_x, preset_y, preset_z = preset_bounds(args.bounds_preset)
    x_bounds = tuple(args.x_bounds) if args.x_bounds is not None else preset_x
    y_bounds = tuple(args.y_bounds) if args.y_bounds is not None else preset_y
    z_bounds = tuple(args.z_bounds) if args.z_bounds is not None else preset_z
    return x_bounds, y_bounds, z_bounds


def evaluate_grid(model, z, x_bounds, y_bounds, z_bounds, resolution, chunk_size, device):
    xs = np.linspace(x_bounds[0], x_bounds[1], resolution, dtype=np.float32)
    ys = np.linspace(y_bounds[0], y_bounds[1], resolution, dtype=np.float32)
    zs = np.linspace(z_bounds[0], z_bounds[1], resolution, dtype=np.float32)

    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    values = []
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk_size):
            end = min(start + chunk_size, points.shape[0])
            p = torch.from_numpy(points[start:end]).float().to(device).unsqueeze(0)
            pred = model(z, p)
            values.append(pred.squeeze(0).squeeze(-1).detach().cpu().numpy())

    grid = np.concatenate(values, axis=0).reshape(resolution, resolution, resolution)
    return grid, xs, ys, zs


def keep_largest_connected_component(faces):
    if faces is None or len(faces) == 0:
        return faces, 0, 0

    faces = np.asarray(faces, dtype=np.int64)
    vertex_to_faces = {}
    for face_index, tri in enumerate(faces):
        for vertex_index in tri:
            vertex_to_faces.setdefault(int(vertex_index), []).append(face_index)

    visited = np.zeros(len(faces), dtype=bool)
    components = []

    for start_face in range(len(faces)):
        if visited[start_face]:
            continue

        stack = [start_face]
        visited[start_face] = True
        component = []

        while stack:
            face_index = stack.pop()
            component.append(face_index)
            for vertex_index in faces[face_index]:
                for neighbor_face in vertex_to_faces.get(int(vertex_index), []):
                    if not visited[neighbor_face]:
                        visited[neighbor_face] = True
                        stack.append(neighbor_face)

        components.append(component)

    if not components:
        return faces, 0, 0

    largest = max(components, key=len)
    keep = np.zeros(len(faces), dtype=bool)
    keep[np.asarray(largest, dtype=np.int64)] = True
    return faces[keep], len(components), len(largest)


def compact_mesh_vertices(vertices, faces, normals=None):
    if faces is None or len(faces) == 0:
        empty_normals = None
        if normals is not None and len(normals) == len(vertices):
            empty_normals = normals[:0]
        return vertices[:0], np.zeros((0, 3), dtype=np.int64), empty_normals

    faces = np.asarray(faces, dtype=np.int64)
    used_vertices = np.unique(faces.reshape(-1))
    remap = np.empty(len(vertices), dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    compact_vertices = vertices[used_vertices]
    compact_faces = remap[faces]

    compact_normals = normals
    if normals is not None and len(normals) == len(vertices):
        compact_normals = normals[used_vertices]

    return compact_vertices, compact_faces, compact_normals


def save_point_cloud_ply(points, normals, output_path):
    has_normals = normals is not None and len(normals) == len(points)
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_normals:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("end_header\n")
        if has_normals:
            for p, n in zip(points, normals):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        else:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    runtime = build_runtime_args(args, config)

    encoder = build_image_encoder_from_args(runtime).to(device)
    occupancy_model = DRMSDFModel(
        latent_dim=runtime.latent_dim,
        point_dim=3,
        hidden_dim=runtime.sdf_hidden_dim,
        num_layers=runtime.sdf_num_layers,
        pe_L=runtime.pe_L,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state"])
    occupancy_state = checkpoint.get("occupancy_state", checkpoint.get("drm_state"))
    if occupancy_state is None:
        raise RuntimeError("Checkpoint is missing occupancy_state/drm_state.")
    occupancy_model.load_state_dict(occupancy_state)
    encoder.eval()
    occupancy_model.eval()

    image = load_conditioning_image(
        args.image_path,
        image_width=runtime.img_width,
        image_height=runtime.img_height,
        image_channels=runtime.image_channels,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        latent_z = encoder(image).unsqueeze(1)

    x_bounds, y_bounds, z_bounds = resolve_bounds(args)
    print(f"Bounds: x={x_bounds} y={y_bounds} z={z_bounds}")
    logits, xs, ys, zs = evaluate_grid(
        model=occupancy_model,
        z=latent_z,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        resolution=args.resolution,
        chunk_size=args.chunk_size,
        device=device,
    )

    if args.use_probability_grid:
        grid = 1.0 / (1.0 + np.exp(-logits))
        iso = args.iso_probability
        field_name = "probability"
    else:
        grid = logits
        iso = args.iso_logit
        field_name = "logit"

    grid_min = float(grid.min())
    grid_max = float(grid.max())
    print(f"Occupancy {field_name} range: [{grid_min:.6f}, {grid_max:.6f}], iso={iso:.6f}")
    if not (grid_min <= iso <= grid_max):
        if not args.allow_iso_fallback:
            raise RuntimeError(
                f"iso={iso:.6f} is outside the {field_name} range. "
                "The model predicts all-inside or all-outside in these bounds; "
                "try better bounds, a later checkpoint, or --allow_iso_fallback."
            )
        iso = 0.5 * (grid_min + grid_max)
        print(f"Using fallback iso={iso:.6f}")

    verts_zyx, faces, normals, _ = measure.marching_cubes(
        grid,
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

    eps = float(args.boundary_eps)
    if eps > 0 and len(faces) > 0:
        before_boundary_faces = len(faces)
        mins = np.array([x_bounds[0], y_bounds[0], z_bounds[0]])
        maxs = np.array([x_bounds[1], y_bounds[1], z_bounds[1]])
        near_min = np.abs(verts_xyz - mins[None, :]) < eps
        near_max = np.abs(verts_xyz - maxs[None, :]) < eps
        near_boundary = np.any(near_min | near_max, axis=1)
        faces = faces[~np.any(near_boundary[faces], axis=1)]
        print(f"Boundary filter: faces {before_boundary_faces} -> {len(faces)}")

    if args.keep_largest_component and len(faces) > 0:
        before_component_faces = len(faces)
        faces, component_count, largest_faces = keep_largest_connected_component(faces)
        print(
            f"Connected components: {component_count}; "
            f"faces {before_component_faces} -> {len(faces)} "
            f"(largest={largest_faces})"
        )

    verts_xyz, faces, normals = compact_mesh_vertices(verts_xyz, faces, normals)
    if len(faces) == 0:
        raise RuntimeError("No mesh faces remain after reconstruction cleanup.")

    os.makedirs(os.path.dirname(args.output_obj) or ".", exist_ok=True)
    save_obj(args.output_obj, verts_xyz, faces)
    print(f"Wrote mesh to: {args.output_obj}")
    print(f"Image: {args.image_path}")
    print(f"Vertices={len(verts_xyz)} Faces={len(faces)}")

    if not args.no_point_cloud:
        pc_path = args.output_point_cloud.strip()
        if not pc_path:
            pc_path = os.path.splitext(args.output_obj)[0] + ".ply"
        os.makedirs(os.path.dirname(pc_path) or ".", exist_ok=True)
        save_point_cloud_ply(verts_xyz, normals, pc_path)
        print(f"Wrote point cloud to: {pc_path}")


if __name__ == "__main__":
    main()

