"""Shared validation helpers for DRM metric scripts."""

import csv
import math
import os

import numpy as np
import torch

from sdf_metrics import (
    evaluate_sdf,
    format_metrics_table,
    is_better_metrics,
    load_mesh_vertices_faces,
    log_metrics_to_wandb,
    mesh_metrics_from_arrays,
)
from tools_2.reconstruct_drm_only_mesh import evaluate_sdf_grid

try:
    from skimage import measure
except Exception:  # pragma: no cover
    measure = None


METRIC_FIELDS = [
    "model",
    "near_surface_mae",
    "sign_accuracy",
    "chamfer_distance",
    "f_score",
    "normal_consistency",
]


def add_mesh_eval_args(parser):
    parser.add_argument("--compute_mesh_metrics", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mesh_eval_count", type=int, default=10, help="Max samples for OBJ/mesh metrics; <=0 means all")
    parser.add_argument("--mesh_resolution", type=int, default=96)
    parser.add_argument("--mesh_chunk_size", type=int, default=65536)
    parser.add_argument("--mesh_iso_level", type=float, default=0.0)
    parser.add_argument("--mesh_padding", type=float, default=0.10)
    parser.add_argument("--mesh_bounds_source", type=str, default="sample", choices=["sample", "human", "human_tight", "cube"])
    parser.add_argument("--x_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--y_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--z_bounds", type=float, nargs=2, default=None)
    parser.add_argument("--boundary_eps", type=float, default=0.03)
    parser.add_argument("--keep_largest_component", dest="keep_largest_component", action="store_true", default=True)
    parser.add_argument("--no_keep_largest_component", dest="keep_largest_component", action="store_false")
    parser.add_argument("--gt_obj_root", type=str, default="", help="Default: dataroot/obj/datalist")
    parser.add_argument("--auto_generate_gt_mesh", type=int, default=1, choices=[0, 1])
    parser.add_argument("--pred_mesh_dir", type=str, default="", help="Optional directory to save predicted OBJ meshes")
    parser.add_argument("--mesh_random_seed", type=int, default=2026)
    parser.add_argument("--nearest_chunk_size", type=int, default=4096)


def init_wandb(args, run_name):
    if getattr(args, "wandb_mode", "disabled") == "disabled" or not getattr(args, "wandb_project", ""):
        return None
    try:
        import wandb
        return wandb.init(
            project=args.wandb_project,
            name=getattr(args, "wandb_run_name", "") or run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )
    except Exception as exc:
        print(f"wandb init failed; continuing without wandb: {exc}")
        return None


def write_csv(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_FIELDS})


def nanmean(values):
    finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def preset_bounds(name):
    if name == "cube":
        return (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)
    if name == "human_tight":
        return (-0.65, 0.65), (-1.05, 1.05), (-0.65, 0.65)
    return (-0.8, 0.8), (-1.15, 1.15), (-0.8, 0.8)


def tensor_item_points(value, item_index):
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        item = value[item_index]
        if item.numel() > 0:
            return item.detach().cpu().numpy().reshape(-1, 3)
    return None


def resolve_mesh_bounds(args, surface_points=None, sdf_points=None):
    if args.x_bounds is not None and args.y_bounds is not None and args.z_bounds is not None:
        return tuple(args.x_bounds), tuple(args.y_bounds), tuple(args.z_bounds)
    if getattr(args, "mesh_bounds_source", "sample") == "sample":
        pts = surface_points if surface_points is not None and len(surface_points) > 0 else sdf_points
        if pts is not None and len(pts) > 0:
            mins = np.min(pts, axis=0) - float(args.mesh_padding)
            maxs = np.max(pts, axis=0) + float(args.mesh_padding)
            return (float(mins[0]), float(maxs[0])), (float(mins[1]), float(maxs[1])), (float(mins[2]), float(maxs[2]))
    return preset_bounds(getattr(args, "mesh_bounds_source", "human"))


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


def compact_mesh_vertices(vertices, faces):
    if faces is None or len(faces) == 0:
        return vertices[:0], np.zeros((0, 3), dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    used_vertices = np.unique(faces.reshape(-1))
    remap = np.empty(len(vertices), dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    return vertices[used_vertices], remap[faces]


def save_obj(path, vertices, faces):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def extract_drm_mesh_arrays(drm, z, args, device, surface_points=None, sdf_points=None):
    if measure is None:
        raise RuntimeError("scikit-image is required for marching cubes")
    x_bounds, y_bounds, z_bounds = resolve_mesh_bounds(args, surface_points=surface_points, sdf_points=sdf_points)
    sdf, xs, ys, zs = evaluate_sdf_grid(
        model=drm,
        z=z,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        resolution=int(args.mesh_resolution),
        chunk_size=int(args.mesh_chunk_size),
        device=device,
    )
    iso = float(args.mesh_iso_level)
    if not (float(np.nanmin(sdf)) <= iso <= float(np.nanmax(sdf))):
        raise RuntimeError(f"iso={iso} outside predicted SDF range [{float(np.nanmin(sdf)):.6g}, {float(np.nanmax(sdf)):.6g}]")
    verts_zyx, faces, _, _ = measure.marching_cubes(
        sdf,
        level=iso,
        spacing=(zs[1] - zs[0], ys[1] - ys[0], xs[1] - xs[0]),
    )
    vertices = np.stack([verts_zyx[:, 2] + xs[0], verts_zyx[:, 1] + ys[0], verts_zyx[:, 0] + zs[0]], axis=1).astype(np.float32)
    faces = faces.astype(np.int64)
    eps = float(getattr(args, "boundary_eps", 0.0))
    if eps > 0 and len(faces) > 0:
        mins = np.array([x_bounds[0], y_bounds[0], z_bounds[0]], dtype=np.float32)
        maxs = np.array([x_bounds[1], y_bounds[1], z_bounds[1]], dtype=np.float32)
        near_boundary = np.any((np.abs(vertices - mins[None, :]) < eps) | (np.abs(vertices - maxs[None, :]) < eps), axis=1)
        faces = faces[~np.any(near_boundary[faces], axis=1)]
    if getattr(args, "keep_largest_component", True) and len(faces) > 0:
        faces, _, _ = keep_largest_connected_component(faces)
    vertices, faces = compact_mesh_vertices(vertices, faces)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("predicted mesh is empty after filtering")
    return vertices, faces


def gt_obj_path_for_sample(args, sample_name):
    gt_root = getattr(args, "gt_obj_root", "") or os.path.join(args.dataroot, "obj", args.datalist)
    return os.path.join(gt_root, sample_name.replace(".png", ".obj"))


def ensure_gt_mesh(args, sample_name):
    obj_path = gt_obj_path_for_sample(args, sample_name)
    if os.path.exists(obj_path):
        return obj_path, ""
    if not getattr(args, "auto_generate_gt_mesh", 1):
        return "", f"missing GT OBJ: {obj_path}"
    npz_path = os.path.join(args.dataroot, "sdf", args.datalist, sample_name.replace(".png", ".npz"))
    if not os.path.exists(npz_path):
        return "", f"missing GT OBJ and SDF NPZ: {obj_path}; {npz_path}"
    try:
        from tools.npz_to_obj_ball_pivoting import reconstruct_mesh
        reconstruct_mesh(npz_path, obj_path)
    except Exception as exc:
        return "", f"failed to generate GT OBJ from {npz_path}: {exc}"
    if not os.path.exists(obj_path):
        return "", f"GT OBJ generation did not create file: {obj_path}"
    return obj_path, ""


def maybe_add_mesh_metric(meter, drm, z, args, device, sample_name, surface_points=None, sdf_points=None):
    if not getattr(args, "compute_mesh_metrics", 0):
        return
    max_count = int(getattr(args, "mesh_eval_count", 10))
    if max_count > 0 and meter.mesh_eval_done >= max_count:
        return
    gt_path, gt_error = ensure_gt_mesh(args, sample_name)
    if gt_error:
        meter.add_mesh_skip(sample_name, gt_error)
        return
    gt_vertices, gt_faces, gt_load_error = load_mesh_vertices_faces(gt_path)
    if gt_load_error:
        meter.add_mesh_skip(sample_name, gt_load_error)
        return
    try:
        pred_vertices, pred_faces = extract_drm_mesh_arrays(
            drm,
            z,
            args,
            device,
            surface_points=surface_points,
            sdf_points=sdf_points,
        )
    except Exception as exc:
        meter.add_mesh_skip(sample_name, f"prediction mesh failed: {exc}")
        return
    if getattr(args, "pred_mesh_dir", ""):
        pred_path = os.path.join(args.pred_mesh_dir, meter.model_name, sample_name.replace(".png", ".obj"))
        save_obj(pred_path, pred_vertices, pred_faces)
    metrics, details = mesh_metrics_from_arrays(
        pred_vertices,
        pred_faces,
        gt_vertices,
        gt_faces,
        delta=float(args.delta),
        num_surface_points=int(args.num_surface_points),
        random_seed=int(getattr(args, "mesh_random_seed", 2026)) + meter.mesh_eval_done,
        nearest_chunk_size=int(getattr(args, "nearest_chunk_size", 4096)),
    )
    meter.add_mesh_metrics(metrics, details, sample_name)


class RunningSDFMetrics:
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.args = args
        self.near_abs_sum = 0.0
        self.near_count = 0
        self.sign_correct = 0.0
        self.total_count = 0
        self.mesh_metric_values = {
            "chamfer_distance": [],
            "f_score": [],
            "normal_consistency": [],
        }
        self.mesh_warning_printed = False
        self.mesh_eval_done = 0
        self.mesh_eval_skipped = 0
        self.seen_batches = 0
        self.valid_batches = 0
        self.empty_batches = 0
        self.skipped_samples = 0
        self.gt_min = float("inf")
        self.gt_max = float("-inf")

    def add_empty_batch(self):
        self.empty_batches += 1

    def add_skipped_samples(self, count):
        self.skipped_samples += int(count)

    def add_mesh_skip(self, sample_name, reason):
        self.mesh_eval_skipped += 1
        if not self.mesh_warning_printed:
            print(f"[{self.model_name}] mesh metric skipped for {sample_name}: {reason}")
            self.mesh_warning_printed = True

    def add_mesh_metrics(self, metrics, details, sample_name):
        for key in self.mesh_metric_values:
            self.mesh_metric_values[key].append(metrics[key])
        self.mesh_eval_done += 1
        if details and not self.mesh_warning_printed:
            print(f"[{self.model_name}] mesh metric details for {sample_name}: {details}")
            self.mesh_warning_printed = True

    def add_batch(self, pred_sdf, gt_sdf, points=None):
        self.valid_batches += 1
        pred_flat = pred_sdf.reshape(-1)
        gt_flat = gt_sdf.reshape(-1)
        if gt_flat.numel() == 0:
            return
        self.gt_min = min(self.gt_min, float(gt_flat.min().detach().item()))
        self.gt_max = max(self.gt_max, float(gt_flat.max().detach().item()))

        near_mask = torch.abs(gt_flat) < float(self.args.tau)
        if near_mask.any():
            self.near_abs_sum += torch.abs(pred_flat[near_mask] - gt_flat[near_mask]).sum().item()
            self.near_count += int(near_mask.sum().item())

        self.sign_correct += ((pred_flat < 0) == (gt_flat < 0)).float().sum().item()
        self.total_count += int(gt_flat.numel())

        if getattr(self.args, "grid_shape", None) is not None:
            batch_size = pred_sdf.shape[0]
            for item_index in range(batch_size):
                metrics, details = evaluate_sdf(
                    pred_sdf[item_index].detach().cpu(),
                    gt_sdf[item_index].detach().cpu(),
                    points=points[item_index].detach().cpu() if isinstance(points, torch.Tensor) else None,
                    grid_shape=self.args.grid_shape,
                    tau=self.args.tau,
                    delta=self.args.delta,
                    num_surface_points=self.args.num_surface_points,
                    return_details=True,
                )
                for key in self.mesh_metric_values:
                    self.mesh_metric_values[key].append(metrics[key])
                if not self.mesh_warning_printed and "mesh_metrics_error" in details:
                    print(f"[{self.model_name}] mesh metrics warning: {details['mesh_metrics_error']}")
                    self.mesh_warning_printed = True

    def finalize(self):
        metrics = {
            "near_surface_mae": self.near_abs_sum / self.near_count if self.near_count > 0 else float("nan"),
            "sign_accuracy": self.sign_correct / self.total_count if self.total_count > 0 else float("nan"),
            "chamfer_distance": nanmean(self.mesh_metric_values["chamfer_distance"]),
            "f_score": nanmean(self.mesh_metric_values["f_score"]),
            "normal_consistency": nanmean(self.mesh_metric_values["normal_consistency"]),
        }
        gt_range = "empty" if self.total_count == 0 else f"[{self.gt_min:.6g}, {self.gt_max:.6g}]"
        print(
            f"[{self.model_name}] validation diagnostics: "
            f"seen_batches={self.seen_batches}, valid_batches={self.valid_batches}, "
            f"empty_batches={self.empty_batches}, skipped_samples={self.skipped_samples}, "
            f"total_sdf_points={self.total_count}, near_surface_points={self.near_count}, "
            f"gt_sdf_range={gt_range}, mesh_eval_done={self.mesh_eval_done}, "
            f"mesh_eval_skipped={self.mesh_eval_skipped}"
        )
        if self.total_count == 0:
            print(f"[{self.model_name}] no SDF validation points were evaluated.")
        if self.near_count == 0 and self.total_count > 0:
            print(f"[{self.model_name}] no near-surface points with tau={self.args.tau}; try --tau 0.05")
        if not getattr(self.args, "compute_mesh_metrics", 0) and getattr(self.args, "grid_shape", None) is None:
            print(f"[{self.model_name}] mesh metrics disabled: pass --compute_mesh_metrics 1 to compare predicted OBJ against GT OBJ")
        return {"model": self.model_name, **metrics}


def print_and_save_results(rows, args, wandb_run=None):
    print(format_metrics_table(rows))
    best_row = None
    for row in rows:
        if is_better_metrics(row, best_row):
            best_row = row
        if wandb_run is not None:
            log_metrics_to_wandb({k: row[k] for k in row if k != "model"}, prefix=f"val/{row['model']}", wandb_run=wandb_run)
    if best_row is not None:
        print(f"Best model by priority: {best_row['model']}")
    write_csv(getattr(args, "output_csv", ""), rows)
    if getattr(args, "output_csv", ""):
        print(f"Wrote metrics CSV: {args.output_csv}")
    if wandb_run is not None:
        wandb_run.finish()
