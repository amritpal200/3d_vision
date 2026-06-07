
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 tools_2_mtm_z/reconstruct_from_mtm_z.py --checkpoint /data/125-1/users/asingh/proves/include_mtm/aligned/MTM_z_for_image_DRM/epoch_64_net_MTMZImageDRM.pth --dataroot /data/113-1/users/asingh/project/3d/MPV3D --datalist test_pairs --datamode aligned --sample_index 0 --output_obj mesh_results/mtm_z_person.obj --output_point_cloud mesh_results/mtm_z_person.ply  --resolution 96 --bounds_preset human
#!/usr/bin/env python3
"""Reconstruct a mesh from MTM z_proj + frozen DRM checkpoint."""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
IMAGE_ENCODER_DIR = os.path.join(PROJECT_ROOT, "tools_2_image_encoder")
for path in (PROJECT_ROOT, CURRENT_DIR, IMAGE_ENCODER_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from tools_2_mtm_z.common import build_mtm, create_mtm_dataset, get_device, prepare_mtm_inputs  # noqa: E402
from tools_2_image_encoder.common import load_conditioning_image  # noqa: E402
from tools_2_image_encoder.image_encoder_model import build_image_encoder_from_args  # noqa: E402
from models_2 import DRMSDFModel  # noqa: E402
from tools_2.reconstruct_drm_only_mesh import evaluate_sdf_grid, save_obj  # noqa: E402
from tools_2_image_encoder.reconstruct_from_image import (  # noqa: E402
    compact_mesh_vertices,
    keep_largest_connected_component,
    preset_bounds,
    save_point_cloud_ply,
)

try:
    from skimage import measure
except ImportError as exc:
    raise ImportError('scikit-image is required for marching cubes. Install with "pip install scikit-image".') from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataroot", type=str, default="mpv3d_example")
    parser.add_argument("--datalist", type=str, default="test_pairs")
    parser.add_argument("--datamode", type=str, default="aligned")
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help=(
            "Conditioning RGB image. Its filename must exist in --datalist so "
            "the matching MTM agnostic and cloth inputs can be loaded."
        ),
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_obj", type=str, default="mesh_results/mtm_z_reconstruction.obj")
    parser.add_argument("--output_point_cloud", type=str, default="")
    parser.add_argument("--no_point_cloud", action="store_true")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--iso_level", type=float, default=0.0)
    parser.add_argument("--bounds_preset", type=str, default="human", choices=["human", "human_tight", "cube"])
    parser.add_argument("--x_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--y_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--z_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--boundary_eps", type=float, default=0.03)
    parser.add_argument("--keep_largest_component", dest="keep_largest_component", action="store_true", default=True)
    parser.add_argument("--no_keep_largest_component", dest="keep_largest_component", action="store_false")
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mtm_fusion_mode", type=str, default="", choices=["", "add", "replace", "concat"])
    parser.add_argument("--mtm_z_scale", type=float, default=None)
    return parser.parse_args()


def fuse_latents(z_image, z_mtm, config):
    mode = config.get("mtm_fusion_mode", "add")
    scale = float(config.get("mtm_z_scale", 1.0))
    scaled_mtm = scale * z_mtm
    if mode == "replace":
        return scaled_mtm
    if mode == "concat":
        return torch.cat([z_image, scaled_mtm], dim=-1)
    return z_image + scaled_mtm


def image_encoder_runtime(config):
    image_config = config.get("image_drm_config", {})
    image_latent_dim = int(config.get("image_latent_dim", image_config.get("latent_dim", config["latent_dim"])))
    return SimpleNamespace(
        latent_dim=image_latent_dim,
        img_width=int(image_config.get("img_width", 320)),
        img_height=int(image_config.get("img_height", 512)),
        image_channels=int(image_config.get("image_channels", 3)),
        encoder_base_channels=int(image_config.get("encoder_base_channels", 32)),
        encoder_num_blocks=int(image_config.get("encoder_num_blocks", 5)),
        encoder_head_hidden_dim=int(image_config.get("encoder_head_hidden_dim", 512)),
        encoder_dropout=float(image_config.get("encoder_dropout", 0.0)),
        encoder_use_batchnorm=int(image_config.get("encoder_use_batchnorm", 1)),
    )


def resolve_bounds(args):
    px, py, pz = preset_bounds(args.bounds_preset)
    return (
        tuple(args.x_bounds) if args.x_bounds is not None else px,
        tuple(args.y_bounds) if args.y_bounds is not None else py,
        tuple(args.z_bounds) if args.z_bounds is not None else pz,
    )


def resolve_sample_index(args, base_dataset):
    if not args.image_path:
        return int(args.sample_index)
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    image_name = os.path.basename(args.image_path)
    sample_names = list(getattr(base_dataset, "im_names", []))
    if image_name not in sample_names:
        raise ValueError(
            f'Image "{image_name}" was not found in {args.dataroot}/{args.datalist}.txt. '
            "MTM reconstruction needs the matching datalist entry to load its "
            "agnostic and cloth inputs."
        )
    sample_index = sample_names.index(image_name)
    print(f"Resolved image {image_name} to sample_index={sample_index}")
    return sample_index


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    if args.mtm_fusion_mode:
        config["mtm_fusion_mode"] = args.mtm_fusion_mode
    if args.mtm_z_scale is not None:
        config["mtm_z_scale"] = args.mtm_z_scale
    image_config = config.get("image_drm_config", {})
    image_latent_dim = int(config.get("image_latent_dim", image_config.get("latent_dim", config["latent_dim"])))
    mtm_latent_dim = int(config.get("mtm_latent_dim", image_latent_dim))
    drm_latent_dim = int(config.get("drm_latent_dim", config.get("latent_dim", image_latent_dim)))
    runtime = SimpleNamespace(
        latent_dim=drm_latent_dim,
        image_latent_dim=image_latent_dim,
        mtm_latent_dim=mtm_latent_dim,
        sdf_hidden_dim=int(config["sdf_hidden_dim"]),
        sdf_num_layers=int(config["sdf_num_layers"]),
        pe_L=int(config["pe_L"]),
    )
    for key, default in (
        ("mtm_input_nc_A", 29),
        ("mtm_input_nc_B", 3),
        ("mtm_ngf", 64),
        ("mtm_n_layers_feat_extract", 3),
        ("mtm_grid_size", 3),
        ("mtm_add_tps", 0),
        ("mtm_add_depth", 0),
        ("mtm_add_segmt", 0),
        ("mtm_norm", "instance"),
        ("mtm_use_dropout", 0),
        ("mtm_init_type", "normal"),
        ("mtm_init_gain", 0.02),
    ):
        setattr(args, key, config.get(key, default))

    mtm = build_mtm(args, runtime.mtm_latent_dim, device)
    mtm.load_state_dict(checkpoint["mtm_state"], strict=False)
    encoder_runtime = image_encoder_runtime(config)
    encoder = build_image_encoder_from_args(encoder_runtime).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    drm = DRMSDFModel(
        latent_dim=runtime.latent_dim,
        point_dim=3,
        hidden_dim=runtime.sdf_hidden_dim,
        num_layers=runtime.sdf_num_layers,
        pe_L=runtime.pe_L,
    ).to(device)
    drm.load_state_dict(checkpoint["drm_state"])
    mtm.eval()
    encoder.eval()
    drm.eval()

    dataset, base_dataset = create_mtm_dataset(args, is_train=True, serial_batches=True)
    sample_index = resolve_sample_index(args, base_dataset)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index must be in [0, {len(dataset) - 1}], got {sample_index}")
    sample = dataset[sample_index]
    if sample is None:
        raise RuntimeError(f"Sample {sample_index} could not be loaded.")
    batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
    mtm_inputs = prepare_mtm_inputs(batch, device)
    if mtm_inputs is None:
        raise RuntimeError("Sample does not contain MTM inputs.")
    agnostic, cloth = mtm_inputs
    if args.image_path:
        image = load_conditioning_image(
            args.image_path,
            image_width=encoder_runtime.img_width,
            image_height=encoder_runtime.img_height,
            image_channels=encoder_runtime.image_channels,
        ).unsqueeze(0).to(device)
    else:
        image = batch.get("conditioning_image", batch.get("person"))
        if not isinstance(image, torch.Tensor):
            raise RuntimeError("Sample does not contain person/conditioning_image for image encoder z.")
        image = image.to(device)
    with torch.no_grad():
        z_mtm = mtm(agnostic, cloth)["z"]
        z_image = encoder(image).unsqueeze(1)
        z = fuse_latents(z_image, z_mtm, config)

    x_bounds, y_bounds, z_bounds = resolve_bounds(args)
    print(f"Bounds: x={x_bounds} y={y_bounds} z={z_bounds}")
    sdf, xs, ys, zs = evaluate_sdf_grid(
        model=drm,
        z=z,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        resolution=args.resolution,
        chunk_size=args.chunk_size,
        device=device,
    )
    iso = args.iso_level
    if not (float(sdf.min()) <= iso <= float(sdf.max())):
        iso = 0.5 * (float(sdf.min()) + float(sdf.max()))
        print(f"Using fallback iso={iso:.6f}")

    verts_zyx, faces, normals, _ = measure.marching_cubes(
        sdf,
        level=iso,
        spacing=(zs[1] - zs[0], ys[1] - ys[0], xs[1] - xs[0]),
    )
    verts_xyz = np.stack([verts_zyx[:, 2] + xs[0], verts_zyx[:, 1] + ys[0], verts_zyx[:, 0] + zs[0]], axis=1)

    eps = float(args.boundary_eps)
    if eps > 0 and len(faces) > 0:
        mins = np.array([x_bounds[0], y_bounds[0], z_bounds[0]])
        maxs = np.array([x_bounds[1], y_bounds[1], z_bounds[1]])
        near_boundary = np.any((np.abs(verts_xyz - mins[None, :]) < eps) | (np.abs(verts_xyz - maxs[None, :]) < eps), axis=1)
        faces = faces[~np.any(near_boundary[faces], axis=1)]
    if args.keep_largest_component and len(faces) > 0:
        faces, component_count, _ = keep_largest_connected_component(faces)
        print(f"Connected components: {component_count}; kept largest")
    verts_xyz, faces, normals = compact_mesh_vertices(verts_xyz, faces, normals)

    os.makedirs(os.path.dirname(args.output_obj) or ".", exist_ok=True)
    save_obj(args.output_obj, verts_xyz, faces)
    print(f"Wrote mesh to: {args.output_obj}")
    if not args.no_point_cloud:
        pc_path = args.output_point_cloud.strip() or os.path.splitext(args.output_obj)[0] + ".ply"
        os.makedirs(os.path.dirname(pc_path) or ".", exist_ok=True)
        save_point_cloud_ply(verts_xyz, normals, pc_path)
        print(f"Wrote point cloud to: {pc_path}")


if __name__ == "__main__":
    main()

