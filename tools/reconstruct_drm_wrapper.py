#!/usr/bin/env python3

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(".")

from tools_2 import reconstruct_drm_only_mesh as base
from models_2 import DRMSDFModel, LatentCodebook
from data import create_dataset

try:
    from models import networks
except Exception:
    networks = None


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def get_drm_state(ckpt):
    if isinstance(ckpt, dict):
        if "drm_state" in ckpt:
            return ckpt["drm_state"]
        if "model_state" in ckpt:
            return ckpt["model_state"]
    return ckpt


def infer_drm_from_weights(drm_state):
    """
    Infer DRM architecture from actual saved weights, not from config.
    Your checkpoint config is wrong, so this is safer.
    """
    first_w = drm_state["layers.0.weight"]
    hidden_dim = first_w.shape[0]
    input_dim = first_w.shape[1]

    # Detect number of hidden layers
    layer_ids = []
    for k in drm_state.keys():
        if k.startswith("layers.") and k.endswith(".weight"):
            layer_ids.append(int(k.split(".")[1]))
    num_layers = max(layer_ids) + 2

    # Try possible pe_L values
    # input_dim = latent_dim + (3 + 3*2*pe_L)
    possible = []
    for pe_L in range(0, 20):
        xyz_dim = 3 + 3 * 2 * pe_L
        latent_dim = input_dim - xyz_dim
        if latent_dim > 0:
            possible.append((latent_dim, pe_L))

    # Prefer common latent sizes
    preferred_latents = [128, 256, 512, 1024]
    for latent_dim, pe_L in possible:
        if latent_dim in preferred_latents:
            return latent_dim, hidden_dim, num_layers, pe_L

    # Fallback
    latent_dim, pe_L = possible[0]
    return latent_dim, hidden_dim, num_layers, pe_L


def clean_state_dict(state_dict):
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    return state_dict


def load_mtm_checkpoint(mtm, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "mtm_state" in ckpt:
        state = ckpt["mtm_state"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    state = clean_state_dict(state)
    result = mtm.load_state_dict(state, strict=False)
    mtm.to(device)

    print("Loaded MTM checkpoint.")
    if hasattr(result, "missing_keys") and result.missing_keys:
        print("MTM missing keys:", result.missing_keys[:20])
    if hasattr(result, "unexpected_keys") and result.unexpected_keys:
        print("MTM unexpected keys:", result.unexpected_keys[:20])

    return ckpt, result


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
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n"
                )
        else:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def build_mtm_model(args, latent_dim, device):
    if networks is None:
        raise RuntimeError("models.networks is not available.")

    return networks.define_MTM(
        input_nc_A=args.mtm_input_nc_A,
        input_nc_B=args.mtm_input_nc_B,
        ngf=args.mtm_ngf,
        n_layers=args.mtm_n_layers_feat_extract,
        img_height=512,
        img_width=320,
        grid_size=args.mtm_grid_size,
        add_tps=args.mtm_add_tps,
        add_depth=args.mtm_add_depth,
        add_segmt=args.mtm_add_segmt,
        latent_dim=latent_dim,
        norm=args.mtm_norm,
        use_dropout=args.mtm_use_dropout,
        init_type=args.mtm_init_type,
        init_gain=args.mtm_init_gain,
        gpu_ids=[device.index] if device.type == "cuda" else [],
    )


def build_dataset(args, is_train=False):
    ds_opt = SimpleNamespace()
    ds_opt.dataroot = args.dataroot
    ds_opt.datalist = args.datalist
    ds_opt.datamode = args.datamode
    ds_opt.model = "MTM"
    ds_opt.batch_size = 1
    ds_opt.img_width = 320
    ds_opt.img_height = 512
    ds_opt.isTrain = is_train
    ds_opt.max_dataset_size = float("inf")
    ds_opt.num_threads = 0
    ds_opt.serial_batches = True
    ds_opt.no_pin_memory = True
    ds_opt.radius = 5
    ds_opt.warproot = ""
    return create_dataset(ds_opt).dataset


def resolve_sample_name(args, sample_names):
    if args.sample_name:
        return args.sample_name

    if sample_names and 0 <= args.sample_index < len(sample_names):
        return sample_names[args.sample_index]

    # Fallback to datalist file when checkpoint has no sample_names.
    try:
        list_path = os.path.join(args.dataroot, args.datalist + ".txt")
        with open(list_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if 0 <= args.sample_index < len(lines):
            im_name, _ = lines[args.sample_index].split()
            return im_name
    except Exception:
        pass

    return ""


def sample_bounds_from_sdf_npz(args, sample_name):
    if not sample_name:
        return None

    sdf_path = os.path.join(args.dataroot, "sdf", args.datalist, sample_name.replace(".png", ".npz"))
    if not os.path.exists(sdf_path):
        return None

    try:
        sdf_npz = np.load(sdf_path)
    except Exception:
        return None

    pts = None
    if "surface_points" in sdf_npz and sdf_npz["surface_points"].size > 0:
        pts = sdf_npz["surface_points"]
    elif "points" in sdf_npz and sdf_npz["points"].size > 0:
        pts = sdf_npz["points"]

    if pts is None or pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None

    mins = pts.min(axis=0) - args.padding
    maxs = pts.max(axis=0) + args.padding
    return (
        (float(mins[0]), float(maxs[0])),
        (float(mins[1]), float(maxs[1])),
        (float(mins[2]), float(maxs[2])),
    )


def is_default_cube_bounds(x_bounds, y_bounds, z_bounds):
    return x_bounds == (-1.2, 1.2) and y_bounds == (-1.2, 1.2) and z_bounds == (-1.2, 1.2)


def keep_largest_connected_component(vertices_xyz, faces):
    if faces is None or len(faces) == 0:
        return faces

    faces = np.asarray(faces, dtype=np.int64)
    n_faces = faces.shape[0]

    # Build vertex -> face adjacency.
    v2f = {}
    for fi, tri in enumerate(faces):
        for v in tri:
            if v not in v2f:
                v2f[v] = []
            v2f[v].append(fi)

    visited = np.zeros(n_faces, dtype=bool)
    components = []

    for start_f in range(n_faces):
        if visited[start_f]:
            continue
        stack = [start_f]
        visited[start_f] = True
        comp = []

        while stack:
            fi = stack.pop()
            comp.append(fi)
            tri = faces[fi]
            for v in tri:
                for nfi in v2f.get(int(v), []):
                    if not visited[nfi]:
                        visited[nfi] = True
                        stack.append(nfi)

        components.append(comp)

    if not components:
        return faces

    largest = max(components, key=len)
    keep = np.zeros(n_faces, dtype=bool)
    keep[np.array(largest, dtype=np.int64)] = True
    return faces[keep]


def refine_bounds_from_sdf_component(model, z, x_bounds, y_bounds, z_bounds, iso_level, device, coarse_resolution=48, chunk_size=131072, margin_voxels=2, fallback_xy_span=1.2):
    sdf, xs, ys, zs = base.evaluate_sdf_grid(
        model=model,
        z=z,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        resolution=coarse_resolution,
        chunk_size=chunk_size,
        device=device,
    )

    sdf_min = float(np.min(sdf))
    sdf_max = float(np.max(sdf))
    sdf_span = max(sdf_max - sdf_min, 1e-6)

    # Prefer a thin near-surface shell to avoid giant connected interior volumes.
    band = max(0.02, 0.03 * sdf_span)
    inside = np.abs(sdf - iso_level) <= band
    if not np.any(inside):
        # Fallback when near-surface mask is empty.
        inside = sdf <= iso_level
    if not np.any(inside):
        return None

    visited = np.zeros_like(inside, dtype=bool)
    nz, ny, nx = inside.shape
    center = np.array([nz // 2, ny // 2, nx // 2], dtype=np.int64)
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    components = []
    active = np.argwhere(inside)

    for seed in active:
        sz, sy, sx = int(seed[0]), int(seed[1]), int(seed[2])
        if visited[sz, sy, sx]:
            continue

        stack = [(sz, sy, sx)]
        visited[sz, sy, sx] = True

        count = 0
        sum_idx = np.zeros(3, dtype=np.float64)
        min_idx = np.array([sz, sy, sx], dtype=np.int64)
        max_idx = np.array([sz, sy, sx], dtype=np.int64)
        contains_center = False

        while stack:
            cz, cy, cx = stack.pop()
            count += 1
            sum_idx += np.array([cz, cy, cx], dtype=np.float64)
            min_idx = np.minimum(min_idx, np.array([cz, cy, cx], dtype=np.int64))
            max_idx = np.maximum(max_idx, np.array([cz, cy, cx], dtype=np.int64))

            if cz == center[0] and cy == center[1] and cx == center[2]:
                contains_center = True

            for dz, dy, dx in neighbors:
                nz_i, ny_i, nx_i = cz + dz, cy + dy, cx + dx
                if nz_i < 0 or ny_i < 0 or nx_i < 0 or nz_i >= nz or ny_i >= ny or nx_i >= nx:
                    continue
                if not inside[nz_i, ny_i, nx_i] or visited[nz_i, ny_i, nx_i]:
                    continue
                visited[nz_i, ny_i, nx_i] = True
                stack.append((nz_i, ny_i, nx_i))

        centroid = sum_idx / max(count, 1)
        dist_to_center = float(np.linalg.norm(centroid - center.astype(np.float64)))
        components.append(
            {
                'count': count,
                'min_idx': min_idx,
                'max_idx': max_idx,
                'contains_center': contains_center,
                'dist_to_center': dist_to_center,
                'centroid': centroid,
            }
        )

    if not components:
        return None

    # Prefer a component that is both sizeable and reasonably central.
    def comp_score(c):
        return c['count'] / (1.0 + c['dist_to_center'])

    best = max(components, key=lambda c: (comp_score(c), c['count']))

    min_voxels = max(16, coarse_resolution // 2)
    if best['count'] < min_voxels:
        # If selected component is too tiny, fallback to largest component.
        best = max(components, key=lambda c: c['count'])

    min_idx = np.maximum(best['min_idx'] - margin_voxels, 0)
    max_idx = np.minimum(best['max_idx'] + margin_voxels, np.array([nz - 1, ny - 1, nx - 1], dtype=np.int64))

    rx = (float(xs[min_idx[2]]), float(xs[max_idx[2]]))
    ry = (float(ys[min_idx[1]]), float(ys[max_idx[1]]))
    rz = (float(zs[min_idx[0]]), float(zs[max_idx[0]]))

    # If x/y still cover almost the full cube, crop around selected component center.
    orig_x_span = float(x_bounds[1] - x_bounds[0])
    orig_y_span = float(y_bounds[1] - y_bounds[0])
    cx = float(xs[int(np.clip(round(best['centroid'][2]), 0, len(xs) - 1))])
    cy = float(ys[int(np.clip(round(best['centroid'][1]), 0, len(ys) - 1))])

    if (rx[1] - rx[0]) > 0.9 * orig_x_span:
        half = min(0.5 * fallback_xy_span, 0.5 * orig_x_span)
        rx = (max(float(x_bounds[0]), cx - half), min(float(x_bounds[1]), cx + half))

    if (ry[1] - ry[0]) > 0.9 * orig_y_span:
        half = min(0.5 * fallback_xy_span, 0.5 * orig_y_span)
        ry = (max(float(y_bounds[0]), cy - half), min(float(y_bounds[1]), cy + half))

    if rx[0] >= rx[1] or ry[0] >= ry[1] or rz[0] >= rz[1]:
        return None

    return rx, ry, rz, len(components), best['count']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--drm_ckpt", required=True)
    parser.add_argument("--mtm_ckpt", default="")

    parser.add_argument("--dataroot", type=str, default="mpv3d_example")
    parser.add_argument("--datalist", type=str, default="train_pairs")
    parser.add_argument("--datamode", type=str, default="aligned")

    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sample_name", type=str, default="")

    parser.add_argument("--output_obj", type=str, default="mesh_results/recon.obj")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--iso_level", type=float, default=0.0)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--no_dataset_bounds", action="store_true")
    parser.add_argument("--save_point_cloud", action="store_true")
    parser.add_argument(
        "--largest_component_only",
        action="store_true",
        default=True,
        help="keep only largest connected mesh component",
    )
    parser.add_argument(
        "--disable_auto_tight_bounds",
        action="store_true",
        default=False,
        help="disable SDF-based pre-pass that tightens reconstruction bounds",
    )
    parser.add_argument(
        "--fallback_xy_span",
        type=float,
        default=1.2,
        help="x/y span used when auto-tight bounds still cover almost full cube",
    )

    parser.add_argument("--gpu_id", type=int, default=0)

    # Optional manual overrides
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--sdf_hidden_dim", type=int, default=None)
    parser.add_argument("--sdf_num_layers", type=int, default=None)
    parser.add_argument("--pe_L", type=int, default=None)

    # MTM args
    parser.add_argument("--mtm_input_nc_A", type=int, default=29)
    parser.add_argument("--mtm_input_nc_B", type=int, default=3)
    parser.add_argument("--mtm_ngf", type=int, default=64)
    parser.add_argument("--mtm_n_layers_feat_extract", type=int, default=3)
    parser.add_argument("--mtm_grid_size", type=int, default=3)
    parser.add_argument("--mtm_add_tps", action="store_true", default=True)
    parser.add_argument("--mtm_add_depth", action="store_true", default=True)
    parser.add_argument("--mtm_add_segmt", action="store_true", default=True)
    parser.add_argument("--mtm_norm", type=str, default="instance")
    parser.add_argument("--mtm_use_dropout", action="store_true", default=False)
    parser.add_argument("--mtm_init_type", type=str, default="normal")
    parser.add_argument("--mtm_init_gain", type=float, default=0.02)

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    print(f"Using device: {device}")

    drm_ckpt = torch.load(args.drm_ckpt, map_location="cpu")
    drm_state = clean_state_dict(get_drm_state(drm_ckpt))

    latent_dim, sdf_hidden_dim, sdf_num_layers, pe_L = infer_drm_from_weights(drm_state)

    if args.latent_dim is not None:
        latent_dim = args.latent_dim
    if args.sdf_hidden_dim is not None:
        sdf_hidden_dim = args.sdf_hidden_dim
    if args.sdf_num_layers is not None:
        sdf_num_layers = args.sdf_num_layers
    if args.pe_L is not None:
        pe_L = args.pe_L

    print("Using DRM architecture:")
    print(f"  latent_dim     = {latent_dim}")
    print(f"  hidden_dim     = {sdf_hidden_dim}")
    print(f"  num_layers     = {sdf_num_layers}")
    print(f"  pe_L           = {pe_L}")

    model = DRMSDFModel(
        latent_dim=latent_dim,
        point_dim=3,
        hidden_dim=sdf_hidden_dim,
        num_layers=sdf_num_layers,
        pe_L=pe_L,
    ).to(device)

    model.load_state_dict(drm_state, strict=True)
    model.eval()
    print("Loaded DRM weights successfully.")

    sample_names = drm_ckpt.get("sample_names", []) if isinstance(drm_ckpt, dict) else []

    z = None

    if isinstance(drm_ckpt, dict) and "latent_state" in drm_ckpt:
        latent = LatentCodebook(
            num_embeddings=len(sample_names),
            latent_dim=latent_dim,
        ).to(device)

        latent.load_state_dict(drm_ckpt["latent_state"])
        latent.eval()

        idx = base.select_sample_index(sample_names, args.sample_index, args.sample_name)

        with torch.no_grad():
            z = latent(torch.tensor([idx], dtype=torch.long, device=device)).unsqueeze(1)

        print("Using latent codebook z.")
        print("z shape:", tuple(z.shape))

    elif args.mtm_ckpt:
        print("MTM checkpoint provided; generating z from MTM.")

        mtm = build_mtm_model(args, latent_dim, device)
        load_mtm_checkpoint(mtm, args.mtm_ckpt, device)
        mtm.eval()

        dataset = build_dataset(args, is_train=False)

        idx = args.sample_index
        if args.sample_name:
            idx = base.select_sample_index(
                sample_names or [str(i) for i in range(len(dataset))],
                args.sample_index,
                args.sample_name,
            )

        sample = dataset[idx]

        agnostic = sample.get("agnostic")
        cloth = sample.get("cloth")

        if not isinstance(agnostic, torch.Tensor):
            raise RuntimeError("Dataset sample missing 'agnostic'.")
        if not isinstance(cloth, torch.Tensor):
            raise RuntimeError("Dataset sample missing 'cloth'.")

        agnostic = agnostic.to(device).unsqueeze(0)
        cloth = cloth.to(device).unsqueeze(0)

        with torch.no_grad():
            out = mtm(agnostic, cloth)

        z = out.get("z")
        if z is None:
            raise RuntimeError("MTM forward did not return z.")

        print("Generated z from MTM.")
        print("z shape:", tuple(z.shape))

        if z.shape[-1] != latent_dim:
            raise RuntimeError(
                f"z latent mismatch: MTM produced {z.shape[-1]}, "
                f"but DRM expects {latent_dim}."
            )

    else:
        raise RuntimeError(
            "No latent source found. Provide --mtm_ckpt or use a DRM checkpoint with latent_state."
        )

    x_bounds, y_bounds, z_bounds = (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)

    if not args.no_dataset_bounds:
        sample_name = resolve_sample_name(args, sample_names)
        try:
            dataset = build_dataset(args, is_train=False)
            idx_for_bounds = args.sample_index
            if args.sample_name:
                idx_for_bounds = base.select_sample_index(
                    sample_names or [str(i) for i in range(len(dataset))],
                    args.sample_index,
                    args.sample_name,
                )

            sample = dataset[idx_for_bounds]
            x_bounds, y_bounds, z_bounds = base.sample_bounds_from_dataset(
                sample,
                padding=args.padding,
            )
            print("Using dataset bounds:")
            print("  x:", x_bounds)
            print("  y:", y_bounds)
            print("  z:", z_bounds)

            if is_default_cube_bounds(x_bounds, y_bounds, z_bounds):
                npz_bounds = sample_bounds_from_sdf_npz(args, sample_name)
                if npz_bounds is not None:
                    x_bounds, y_bounds, z_bounds = npz_bounds
                    print(f"Dataset returned default cube; using SDF NPZ bounds for sample: {sample_name}")
                    print("  x:", x_bounds)
                    print("  y:", y_bounds)
                    print("  z:", z_bounds)
        except Exception as exc:
            print("Failed to infer dataset bounds.")
            print("Reason:", exc)

            npz_bounds = sample_bounds_from_sdf_npz(args, sample_name)
            if npz_bounds is not None:
                x_bounds, y_bounds, z_bounds = npz_bounds
                print(f"Using SDF NPZ bounds for sample: {sample_name}")
                print("  x:", x_bounds)
                print("  y:", y_bounds)
                print("  z:", z_bounds)
            else:
                print("SDF NPZ bounds unavailable; using default cube.")

    if not args.disable_auto_tight_bounds:
        refined = refine_bounds_from_sdf_component(
            model=model,
            z=z,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            iso_level=args.iso_level,
            device=device,
            coarse_resolution=min(48, max(24, args.resolution // 2)),
            chunk_size=max(args.chunk_size, 131072),
            margin_voxels=2,
            fallback_xy_span=args.fallback_xy_span,
        )
        if refined is not None:
            x_bounds, y_bounds, z_bounds, n_comp, best_count = refined
            print("Using auto-tightened SDF bounds:")
            print("  x:", x_bounds)
            print("  y:", y_bounds)
            print("  z:", z_bounds)
            print(f"  coarse components={n_comp}, selected_voxels={best_count}")
        else:
            print("Auto-tight bounds could not be estimated; using current bounds.")

    sdf, xs, ys, zs = base.evaluate_sdf_grid(
        model=model,
        z=z,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        resolution=args.resolution,
        chunk_size=args.chunk_size,
        device=device,
    )

    print("SDF grid stats:")
    print("  min:", float(np.min(sdf)))
    print("  max:", float(np.max(sdf)))
    print("  iso_level:", args.iso_level)

    os.makedirs(os.path.dirname(args.output_obj) or ".", exist_ok=True)

    try:
        from skimage import measure

        verts_zyx, faces, normals, _ = measure.marching_cubes(
            sdf,
            level=args.iso_level,
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

    except Exception as exc:
        raise RuntimeError("marching_cubes failed: " + str(exc))

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

    if args.largest_component_only and faces.size > 0:
        faces = keep_largest_connected_component(verts_xyz, faces)
        print("Kept largest connected mesh component.")

    base.save_obj(args.output_obj, verts_xyz, faces)
    print(f"Wrote mesh to: {args.output_obj}")

    if args.save_point_cloud:
        pc_path = os.path.splitext(args.output_obj)[0] + ".ply"
        save_point_cloud_ply(verts_xyz, normals if normals is not None else None, pc_path)
        print(f"Saved point cloud: {pc_path}")


if __name__ == "__main__":
    main()