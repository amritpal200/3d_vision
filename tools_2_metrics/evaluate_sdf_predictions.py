#!/usr/bin/env python3
"""Evaluate one or more saved SDF prediction arrays.

Each input .npz should contain at least:
    pred_sdf: predicted SDF values
    gt_sdf: ground-truth SDF values at the same points

For mesh metrics, also pass --grid_shape and --bounds, or store structured
points under --points_key so bounds can be inferred.
"""

import argparse
import csv
import os
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sdf_metrics import evaluate_sdf, format_metrics_table, log_metrics_to_wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help=".npz files containing pred_sdf and gt_sdf")
    parser.add_argument("--model_names", nargs="*", default=None)
    parser.add_argument("--pred_key", type=str, default="pred_sdf")
    parser.add_argument("--gt_key", type=str, default="gt_sdf")
    parser.add_argument("--points_key", type=str, default="points")
    parser.add_argument("--grid_shape", type=int, nargs=3, default=None, help="Dense grid shape as nz ny nx")
    parser.add_argument("--bounds", type=float, nargs=6, default=None, help="xmin xmax ymin ymax zmin zmax")
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--num_surface_points", type=int, default=100000)
    parser.add_argument("--no_mesh_metrics", action="store_true")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="sdf_metrics_eval")
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def parse_bounds(values):
    if values is None:
        return None
    return ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))


def write_csv(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "model",
        "near_surface_mae",
        "sign_accuracy",
        "chamfer_distance",
        "f_score",
        "normal_consistency",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    args = parse_args()
    if args.model_names and len(args.model_names) != len(args.inputs):
        raise ValueError("--model_names must have the same length as --inputs")

    wandb_run = None
    if args.wandb_mode != "disabled" and args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=vars(args),
            )
        except Exception as exc:
            print(f"wandb init failed; continuing without wandb: {exc}")

    rows = []
    bounds = parse_bounds(args.bounds)
    for index, input_path in enumerate(args.inputs):
        data = np.load(input_path)
        model_name = args.model_names[index] if args.model_names else os.path.splitext(os.path.basename(input_path))[0]
        if args.pred_key not in data or args.gt_key not in data:
            raise KeyError(f"{input_path} must contain {args.pred_key} and {args.gt_key}")
        points = data[args.points_key] if args.points_key in data else None
        metrics, details = evaluate_sdf(
            data[args.pred_key],
            data[args.gt_key],
            points=points,
            grid_shape=args.grid_shape,
            bounds=bounds,
            tau=args.tau,
            delta=args.delta,
            num_surface_points=args.num_surface_points,
            compute_mesh_metrics=not args.no_mesh_metrics,
            return_details=True,
        )
        row = {"model": model_name, **metrics}
        rows.append(row)
        log_metrics_to_wandb(metrics, prefix=f"eval/{model_name}", wandb_run=wandb_run)
        for key, value in details.items():
            if key.endswith("_error"):
                print(f"[{model_name}] {key}: {value}")

    print(format_metrics_table(rows))
    write_csv(args.output_csv, rows)
    if args.output_csv:
        print(f"Wrote metrics CSV: {args.output_csv}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

