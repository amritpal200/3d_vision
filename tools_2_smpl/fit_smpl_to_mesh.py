#!/usr/bin/env python3
"""Fit an SMPL body to an OBJ/PLY mesh or point cloud without training.

This script optimizes SMPL pose, shape, translation, and a global scale so that
SMPL vertices match points sampled from an input reconstruction. It is intended
as a post-processing step:

    DRM OBJ/PLY -> SMPL fitting -> SMPL params .npz/.pkl + fitted SMPL .obj

Requirements:
    pip install smplx trimesh scipy

You also need the official SMPL model file, for example SMPL_NEUTRAL.pkl.
"""

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh", type=str, required=True, help="Input .obj/.ply mesh or point cloud")
    parser.add_argument("--smpl_model_path", type=str, required=True, help="SMPL model directory or SMPL_*.pkl file")
    parser.add_argument("--output_params", type=str, required=True, help="Output .npz file for fitted SMPL parameters")
    parser.add_argument("--output_obj", type=str, default="", help="Optional fitted SMPL .obj output")
    parser.add_argument("--output_pkl", type=str, default="", help="Optional pickle output with the same parameters")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--num_betas", type=int, default=10)
    parser.add_argument("--num_target_points", type=int, default=20000)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--init_iters", type=int, default=200, help="First stage optimizes scale/translation/orient/shape only")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--init_lr", type=float, default=1e-2)
    parser.add_argument("--lambda_pose", type=float, default=1e-4)
    parser.add_argument("--lambda_betas", type=float, default=1e-3)
    parser.add_argument("--lambda_scale", type=float, default=1e-4)
    parser.add_argument("--nearest_chunk_size", type=int, default=4096)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cuda:0 or cpu")
    return parser.parse_args()


def get_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available() and args.gpu_id >= 0:
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cpu")


def require_imports():
    try:
        import trimesh
    except Exception as exc:
        raise ImportError('trimesh is required. Install with: pip install trimesh scipy') from exc
    try:
        import smplx
    except Exception as exc:
        raise ImportError('smplx is required. Install with: pip install smplx') from exc
    return trimesh, smplx


def load_target_points(path, num_points, seed):
    trimesh, _ = require_imports()
    mesh_or_scene = trimesh.load(path, process=False)

    if isinstance(mesh_or_scene, trimesh.Scene):
        geometries = [g for g in mesh_or_scene.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
        if not geometries:
            raise ValueError(f"No geometry found in scene: {path}")
        meshes = [g for g in geometries if hasattr(g, "faces") and len(g.faces) > 0]
        if meshes:
            mesh_or_scene = trimesh.util.concatenate(meshes)
        else:
            vertices = np.concatenate([np.asarray(g.vertices) for g in geometries], axis=0)
            mesh_or_scene = trimesh.points.PointCloud(vertices)

    rng = np.random.default_rng(int(seed))
    vertices = np.asarray(mesh_or_scene.vertices, dtype=np.float32)
    if vertices.size == 0:
        raise ValueError(f"Input has no vertices: {path}")

    if hasattr(mesh_or_scene, "faces") and len(mesh_or_scene.faces) > 0:
        try:
            points, _ = trimesh.sample.sample_surface(mesh_or_scene, int(num_points))
            points = points.astype(np.float32)
        except Exception:
            points = vertices
    else:
        points = vertices

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError("Input contains no finite points")
    if len(points) > num_points:
        indices = rng.choice(len(points), size=int(num_points), replace=False)
        points = points[indices]
    return points.astype(np.float32)


def resolve_smpl_model_path(smpl_model_path, gender):
    path = Path(smpl_model_path)
    if path.is_file():
        return str(path)
    direct_file = path / f"SMPL_{gender.upper()}.pkl"
    if direct_file.exists():
        return str(direct_file)
    return str(path)


def create_smpl_model(smpl_model_path, gender, num_betas, device):
    _, smplx = require_imports()
    model_path = resolve_smpl_model_path(smpl_model_path, gender)
    print(f"Using SMPL model path: {model_path}")
    model = smplx.create(
        model_path=model_path,
        model_type="smpl",
        gender=gender,
        num_betas=int(num_betas),
        batch_size=1,
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_transl=False,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def nearest_squared(source, target, chunk_size):
    chunks = []
    indices = []
    for start in range(0, source.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), source.shape[0])
        dist = torch.cdist(source[start:end], target, p=2) ** 2
        values, idx = dist.min(dim=1)
        chunks.append(values)
        indices.append(idx)
    return torch.cat(chunks, dim=0), torch.cat(indices, dim=0)


def save_obj(path, vertices, faces):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def initialize_params(model, target_points, num_betas, device):
    zeros_betas = torch.zeros(1, int(num_betas), device=device)
    zeros_pose = torch.zeros(1, 69, device=device)
    zeros_orient = torch.zeros(1, 3, device=device)
    with torch.no_grad():
        output = model(betas=zeros_betas, body_pose=zeros_pose, global_orient=zeros_orient, return_verts=True)
        smpl_verts = output.vertices[0]

    target_min = target_points.min(dim=0).values
    target_max = target_points.max(dim=0).values
    smpl_min = smpl_verts.min(dim=0).values
    smpl_max = smpl_verts.max(dim=0).values

    target_extent = target_max - target_min
    smpl_extent = smpl_max - smpl_min
    target_height = torch.max(target_extent)
    smpl_height = torch.max(smpl_extent).clamp_min(1e-6)
    init_scale = (target_height / smpl_height).clamp_min(1e-6)

    target_center = 0.5 * (target_min + target_max)
    smpl_center = 0.5 * (smpl_min + smpl_max)
    init_transl = target_center - init_scale * smpl_center

    return {
        "betas": torch.nn.Parameter(torch.zeros(1, int(num_betas), device=device)),
        "body_pose": torch.nn.Parameter(torch.zeros(1, 69, device=device)),
        "global_orient": torch.nn.Parameter(torch.zeros(1, 3, device=device)),
        "transl": torch.nn.Parameter(init_transl.view(1, 3).detach().clone()),
        "log_scale": torch.nn.Parameter(torch.log(init_scale).view(1).detach().clone()),
    }


def forward_smpl_vertices(model, params):
    output = model(
        betas=params["betas"],
        body_pose=params["body_pose"],
        global_orient=params["global_orient"],
        return_verts=True,
    )
    scale = torch.exp(params["log_scale"]).view(1, 1, 1)
    transl = params["transl"].view(1, 1, 3)
    return scale * output.vertices + transl


def build_optimizer(params, lr, train_pose):
    opt_params = [params["betas"], params["global_orient"], params["transl"], params["log_scale"]]
    if train_pose:
        opt_params.append(params["body_pose"])
    return torch.optim.Adam(opt_params, lr=float(lr))


def fit_smpl(args):
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = get_device(args)
    print(f"Using device: {device}")

    target_np = load_target_points(args.input_mesh, args.num_target_points, args.seed)
    target = torch.from_numpy(target_np).to(device)
    print(f"Loaded target points: {target.shape[0]} from {args.input_mesh}")
    print(
        "Target bounds: "
        f"min={target.min(dim=0).values.detach().cpu().numpy()} "
        f"max={target.max(dim=0).values.detach().cpu().numpy()}"
    )

    model = create_smpl_model(args.smpl_model_path, args.gender, args.num_betas, device)
    params = initialize_params(model, target, args.num_betas, device)
    optimizer = build_optimizer(params, args.init_lr, train_pose=False)

    best_loss = math.inf
    best_state = None
    total_iters = int(args.num_iters)
    init_iters = min(int(args.init_iters), total_iters)

    for step in range(1, total_iters + 1):
        if step == init_iters + 1:
            optimizer = build_optimizer(params, args.lr, train_pose=True)
            print("Starting stage 2: optimizing full SMPL body pose.")

        vertices = forward_smpl_vertices(model, params)[0]
        target_to_smpl, _ = nearest_squared(target, vertices, args.nearest_chunk_size)
        smpl_to_target, _ = nearest_squared(vertices, target, args.nearest_chunk_size)
        chamfer = target_to_smpl.mean() + smpl_to_target.mean()
        pose_reg = (params["body_pose"] ** 2).mean()
        beta_reg = (params["betas"] ** 2).mean()
        scale_reg = (params["log_scale"] ** 2).mean()
        loss = chamfer + args.lambda_pose * pose_reg + args.lambda_betas * beta_reg + args.lambda_scale * scale_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in params.items()}

        if step == 1 or step % max(1, int(args.print_every)) == 0 or step == total_iters:
            print(
                f"step={step}/{total_iters} loss={loss_value:.8f} "
                f"chamfer={float(chamfer.detach().item()):.8f} "
                f"pose={float(pose_reg.detach().item()):.8f} "
                f"betas={float(beta_reg.detach().item()):.8f} "
                f"scale={float(torch.exp(params['log_scale']).detach().item()):.6f}"
            )

    if best_state is None:
        raise RuntimeError("Optimization did not produce a fitted state")
    for key, value in best_state.items():
        params[key].data.copy_(value.to(device))

    with torch.no_grad():
        fitted_vertices = forward_smpl_vertices(model, params)[0].detach().cpu().numpy().astype(np.float32)

    faces = np.asarray(model.faces, dtype=np.int64)
    smpl_params = {
        "global_orient": params["global_orient"].detach().cpu().numpy().astype(np.float32),
        "body_pose": params["body_pose"].detach().cpu().numpy().astype(np.float32),
        "betas": params["betas"].detach().cpu().numpy().astype(np.float32),
        "transl": params["transl"].detach().cpu().numpy().astype(np.float32),
        "scale": np.exp(params["log_scale"].detach().cpu().numpy()).astype(np.float32),
        "vertices": fitted_vertices,
        "faces": faces.astype(np.int64),
        "gender": np.array(args.gender),
        "model_type": np.array("smpl"),
        "smpl_model_path": np.array(args.smpl_model_path),
        "input_mesh": np.array(args.input_mesh),
        "best_loss": np.array(best_loss, dtype=np.float32),
    }

    output_params = args.output_params
    if not output_params.endswith(".npz"):
        fixed_output = output_params + ".npz"
        print(f"Output params path should end with .npz; writing to: {fixed_output}")
        output_params = fixed_output
    os.makedirs(os.path.dirname(output_params) or ".", exist_ok=True)
    np.savez(output_params, **smpl_params)
    print(f"Wrote SMPL params: {output_params}")

    if args.output_pkl:
        os.makedirs(os.path.dirname(args.output_pkl) or ".", exist_ok=True)
        with open(args.output_pkl, "wb") as f:
            pickle.dump(smpl_params, f)
        print(f"Wrote SMPL pickle: {args.output_pkl}")

    if args.output_obj:
        save_obj(args.output_obj, fitted_vertices, faces)
        print(f"Wrote fitted SMPL mesh: {args.output_obj}")


def main():
    args = parse_args()
    try:
        fit_smpl(args)
    except ImportError as exc:
        print(f"Dependency error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
