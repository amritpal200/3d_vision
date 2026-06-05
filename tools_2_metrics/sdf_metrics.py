"""SDF validation metrics for implicit human reconstruction.

The point metrics only require `pred_sdf` and `gt_sdf` evaluated at identical
3D points. Mesh metrics require values on a structured dense grid so marching
cubes can extract a surface.
"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from skimage import measure
except Exception:  # pragma: no cover - optional dependency guard
    measure = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency guard
    cKDTree = None


MetricDict = Dict[str, float]
Bounds = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


@dataclass
class MetricConfig:
    tau: float = 0.01
    delta: float = 0.01
    num_surface_points: int = 100000
    iso_level: float = 0.0
    random_seed: int = 2026
    nearest_chunk_size: int = 4096
    compute_mesh_metrics: bool = True


METRIC_KEYS = [
    "near_surface_mae",
    "sign_accuracy",
    "chamfer_distance",
    "f_score",
    "normal_consistency",
]


def _nan() -> float:
    return float("nan")


def _to_flat_tensor(value, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().float().reshape(-1)
    if value.numel() == 0:
        raise ValueError(f"{name} is empty")
    return value


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_bounds(bounds: Optional[Sequence[Sequence[float]]]) -> Optional[Bounds]:
    if bounds is None:
        return None
    if len(bounds) != 3:
        raise ValueError("bounds must be ((xmin,xmax), (ymin,ymax), (zmin,zmax))")
    parsed = []
    for axis_bounds in bounds:
        if len(axis_bounds) != 2:
            raise ValueError("each bounds axis must contain exactly two values")
        lo, hi = float(axis_bounds[0]), float(axis_bounds[1])
        if hi <= lo:
            raise ValueError(f"invalid bounds axis: {(lo, hi)}")
        parsed.append((lo, hi))
    return parsed[0], parsed[1], parsed[2]


def _infer_bounds_from_points(points, grid_shape) -> Optional[Bounds]:
    if points is None:
        return None
    pts = _to_numpy(points).reshape(-1, 3)
    expected = int(np.prod(grid_shape))
    if pts.shape[0] != expected:
        return None
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return (
        (float(mins[0]), float(maxs[0])),
        (float(mins[1]), float(maxs[1])),
        (float(mins[2]), float(maxs[2])),
    )


def _sdf_to_grid(sdf, grid_shape: Sequence[int]) -> np.ndarray:
    grid_shape = tuple(int(v) for v in grid_shape)
    if len(grid_shape) != 3:
        raise ValueError("grid_shape must be three integers, usually (nz, ny, nx)")
    sdf_np = _to_numpy(sdf).astype(np.float32).reshape(-1)
    expected = int(np.prod(grid_shape))
    if sdf_np.size != expected:
        raise ValueError(f"SDF has {sdf_np.size} values, but grid_shape requires {expected}")
    return sdf_np.reshape(grid_shape)


def _mesh_from_sdf_grid(
    sdf_grid: np.ndarray,
    bounds: Bounds,
    iso_level: float,
):
    if measure is None:
        return None, None, None, "scikit-image is not available"

    sdf_min = float(np.nanmin(sdf_grid))
    sdf_max = float(np.nanmax(sdf_grid))
    if not np.isfinite(sdf_min) or not np.isfinite(sdf_max):
        return None, None, None, "SDF grid contains no finite values"
    if not (sdf_min <= iso_level <= sdf_max):
        return None, None, None, (
            f"iso_level={iso_level} outside SDF range [{sdf_min:.6f}, {sdf_max:.6f}]"
        )

    nz, ny, nx = sdf_grid.shape
    x_bounds, y_bounds, z_bounds = bounds
    dx = (x_bounds[1] - x_bounds[0]) / max(nx - 1, 1)
    dy = (y_bounds[1] - y_bounds[0]) / max(ny - 1, 1)
    dz = (z_bounds[1] - z_bounds[0]) / max(nz - 1, 1)

    try:
        verts_zyx, faces, normals_zyx, _ = measure.marching_cubes(
            sdf_grid,
            level=float(iso_level),
            spacing=(dz, dy, dx),
        )
    except Exception as exc:
        return None, None, None, f"marching cubes failed: {exc}"

    vertices = np.stack(
        [
            verts_zyx[:, 2] + x_bounds[0],
            verts_zyx[:, 1] + y_bounds[0],
            verts_zyx[:, 0] + z_bounds[0],
        ],
        axis=1,
    ).astype(np.float32)

    normals = np.stack(
        [normals_zyx[:, 2], normals_zyx[:, 1], normals_zyx[:, 0]],
        axis=1,
    ).astype(np.float32)
    normals = _normalize_np(normals)

    return vertices, faces.astype(np.int64), normals, ""


def _normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return _normalize_np(normals.astype(np.float32))


def sample_points_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        raise ValueError("cannot sample an empty mesh")

    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 1e-12
    if not np.any(valid):
        raise ValueError("mesh has no non-degenerate faces")

    faces = faces[valid]
    tri = vertices[faces]
    areas = areas[valid]
    probabilities = areas / areas.sum()

    rng = np.random.default_rng(int(seed))
    face_indices = rng.choice(len(faces), size=int(num_points), replace=True, p=probabilities)
    chosen = tri[face_indices]

    u = rng.random((int(num_points), 1), dtype=np.float32)
    v = rng.random((int(num_points), 1), dtype=np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0]) + v * (chosen[:, 2] - chosen[:, 0])

    normals_per_face = _face_normals(vertices, faces)
    normals = normals_per_face[face_indices]
    return points.astype(np.float32), normals.astype(np.float32)


def _nearest_neighbor_squared(
    source: np.ndarray,
    target: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if cKDTree is not None:
        tree = cKDTree(target)
        dist, index = tree.query(source, k=1)
        return (dist.astype(np.float64) ** 2), index.astype(np.int64)

    source_t = torch.from_numpy(source).float()
    target_t = torch.from_numpy(target).float()
    all_min_dist = []
    all_min_idx = []
    for start in range(0, source_t.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), source_t.shape[0])
        dist = torch.cdist(source_t[start:end], target_t, p=2).pow(2)
        min_dist, min_idx = dist.min(dim=1)
        all_min_dist.append(min_dist.cpu().numpy())
        all_min_idx.append(min_idx.cpu().numpy())
    return np.concatenate(all_min_dist), np.concatenate(all_min_idx).astype(np.int64)


def _mesh_metrics(
    pred_grid: np.ndarray,
    gt_grid: np.ndarray,
    bounds: Bounds,
    config: MetricConfig,
) -> Tuple[MetricDict, Dict[str, str]]:
    metrics = {
        "chamfer_distance": _nan(),
        "f_score": _nan(),
        "normal_consistency": _nan(),
    }
    details = {}

    pred_vertices, pred_faces, _, pred_msg = _mesh_from_sdf_grid(pred_grid, bounds, config.iso_level)
    gt_vertices, gt_faces, _, gt_msg = _mesh_from_sdf_grid(gt_grid, bounds, config.iso_level)
    if pred_msg:
        details["pred_mesh_error"] = pred_msg
    if gt_msg:
        details["gt_mesh_error"] = gt_msg
    if pred_msg or gt_msg:
        return metrics, details

    try:
        pred_points, pred_normals = sample_points_from_mesh(
            pred_vertices,
            pred_faces,
            config.num_surface_points,
            config.random_seed,
        )
        gt_points, gt_normals = sample_points_from_mesh(
            gt_vertices,
            gt_faces,
            config.num_surface_points,
            config.random_seed + 17,
        )
    except Exception as exc:
        details["surface_sampling_error"] = str(exc)
        return metrics, details

    pred_to_gt_sq, pred_to_gt_idx = _nearest_neighbor_squared(
        pred_points,
        gt_points,
        config.nearest_chunk_size,
    )
    gt_to_pred_sq, gt_to_pred_idx = _nearest_neighbor_squared(
        gt_points,
        pred_points,
        config.nearest_chunk_size,
    )

    metrics["chamfer_distance"] = float(pred_to_gt_sq.mean() + gt_to_pred_sq.mean())

    delta_sq = float(config.delta) ** 2
    precision = float((pred_to_gt_sq <= delta_sq).mean())
    recall = float((gt_to_pred_sq <= delta_sq).mean())
    if precision + recall > 0:
        metrics["f_score"] = float(2.0 * precision * recall / (precision + recall))
    else:
        metrics["f_score"] = 0.0

    pred_normal_match = np.abs((pred_normals * gt_normals[pred_to_gt_idx]).sum(axis=-1))
    gt_normal_match = np.abs((gt_normals * pred_normals[gt_to_pred_idx]).sum(axis=-1))
    metrics["normal_consistency"] = float(0.5 * (pred_normal_match.mean() + gt_normal_match.mean()))
    details["precision"] = f"{precision:.6f}"
    details["recall"] = f"{recall:.6f}"
    return metrics, details


def evaluate_sdf(
    pred_sdf,
    gt_sdf,
    *,
    points=None,
    grid_shape: Optional[Sequence[int]] = None,
    bounds: Optional[Sequence[Sequence[float]]] = None,
    tau: float = 0.01,
    delta: float = 0.01,
    num_surface_points: int = 100000,
    iso_level: float = 0.0,
    compute_mesh_metrics: bool = True,
    return_details: bool = False,
) -> MetricDict:
    """Evaluate SDF predictions.

    Args:
        pred_sdf: predicted SDF values at query points, any shape.
        gt_sdf: ground-truth SDF values at the same query points, same count.
        points: optional query points. Used to infer bounds for structured grids.
        grid_shape: optional dense-grid shape `(nz, ny, nx)`. Required for mesh metrics.
        bounds: optional grid bounds `((xmin,xmax), (ymin,ymax), (zmin,zmax))`.
        tau: near-surface threshold.
        delta: F-score distance threshold.
        num_surface_points: number of mesh surface points sampled for CD/F-score/NC.
        iso_level: marching-cubes iso level.
        compute_mesh_metrics: if False, skip CD/F-score/NC.
        return_details: if True, return `(metrics, details)`.

    Returns:
        Dict with keys:
        `near_surface_mae`, `sign_accuracy`, `chamfer_distance`, `f_score`,
        `normal_consistency`.
    """

    pred = _to_flat_tensor(pred_sdf, "pred_sdf")
    gt = _to_flat_tensor(gt_sdf, "gt_sdf")
    if pred.numel() != gt.numel():
        raise ValueError(f"pred_sdf has {pred.numel()} values, gt_sdf has {gt.numel()}")

    metrics = {
        "near_surface_mae": _nan(),
        "sign_accuracy": _nan(),
        "chamfer_distance": _nan(),
        "f_score": _nan(),
        "normal_consistency": _nan(),
    }
    details = {
        "num_points": str(int(gt.numel())),
        "num_near_surface_points": "0",
    }

    near_mask = torch.abs(gt) < float(tau)
    near_count = int(near_mask.sum().item())
    details["num_near_surface_points"] = str(near_count)
    if near_count > 0:
        metrics["near_surface_mae"] = float(torch.abs(pred[near_mask] - gt[near_mask]).mean().item())
    else:
        details["near_surface_error"] = f"no gt_sdf points with |gt_sdf| < tau={tau}"

    pred_sign = pred < 0
    gt_sign = gt < 0
    metrics["sign_accuracy"] = float((pred_sign == gt_sign).float().mean().item())

    if compute_mesh_metrics and grid_shape is not None:
        config = MetricConfig(
            tau=float(tau),
            delta=float(delta),
            num_surface_points=int(num_surface_points),
            iso_level=float(iso_level),
        )
        parsed_bounds = _as_bounds(bounds)
        if parsed_bounds is None:
            parsed_bounds = _infer_bounds_from_points(points, grid_shape)
        if parsed_bounds is None:
            details["mesh_metrics_error"] = (
                "mesh metrics need bounds or structured points to infer bounds"
            )
        else:
            try:
                pred_grid = _sdf_to_grid(pred, grid_shape)
                gt_grid = _sdf_to_grid(gt, grid_shape)
                mesh_metrics, mesh_details = _mesh_metrics(pred_grid, gt_grid, parsed_bounds, config)
                metrics.update(mesh_metrics)
                details.update(mesh_details)
            except Exception as exc:
                details["mesh_metrics_error"] = str(exc)
    elif compute_mesh_metrics:
        details["mesh_metrics_error"] = "mesh metrics skipped because grid_shape was not provided"

    if return_details:
        return metrics, details
    return metrics


def log_metrics_to_wandb(
    metrics: MetricDict,
    *,
    prefix: str = "val",
    step: Optional[int] = None,
    wandb_run=None,
):
    payload = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        payload[f"{prefix}/{key}"] = value

    if not payload:
        return

    if wandb_run is not None:
        wandb_run.log(payload, step=step)
        return

    try:
        import wandb

        wandb.log(payload, step=step)
    except Exception:
        return


def _fmt_metric(value: float) -> str:
    if value is None:
        return "nan"
    try:
        if math.isnan(float(value)):
            return "nan"
    except Exception:
        return str(value)
    return f"{float(value):.6g}"


def format_metrics_table(rows: Iterable[Dict[str, object]]) -> str:
    headers = [
        "Model",
        "NearSurfaceMAE ↓",
        "SignAccuracy ↑",
        "ChamferDistance ↓",
        "FScore ↑",
        "NormalConsistency ↑",
    ]
    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted_rows.append(
            [
                str(row.get("model", "")),
                _fmt_metric(row.get("near_surface_mae", _nan())),
                _fmt_metric(row.get("sign_accuracy", _nan())),
                _fmt_metric(row.get("chamfer_distance", _nan())),
                _fmt_metric(row.get("f_score", _nan())),
                _fmt_metric(row.get("normal_consistency", _nan())),
            ]
        )

    widths = [len(h) for h in headers]
    for row in formatted_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def line(cells):
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    sep = "-+-".join("-" * width for width in widths)
    return "\n".join([line(headers), sep] + [line(row) for row in formatted_rows])


def _metric_value_for_sort(metrics: MetricDict, key: str, higher_is_better: bool) -> float:
    value = float(metrics.get(key, _nan()))
    if math.isnan(value):
        return float("inf")
    return -value if higher_is_better else value


def is_better_metrics(candidate: MetricDict, current_best: Optional[MetricDict]) -> bool:
    if current_best is None:
        return True

    priority = [
        ("near_surface_mae", False),
        ("sign_accuracy", True),
        ("chamfer_distance", False),
        ("f_score", True),
        ("normal_consistency", True),
    ]
    for key, higher_is_better in priority:
        cand = _metric_value_for_sort(candidate, key, higher_is_better)
        best = _metric_value_for_sort(current_best, key, higher_is_better)
        if cand < best:
            return True
        if cand > best:
            return False
    return False

