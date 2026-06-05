
# python3 parcial/generate_sdf_dataset.py --input_dir /data/113-1/users/asingh/humanMesh_dataset/input --output_dir /data/113-1/users/asingh/humanMesh_dataset/output --num_points 100000


import argparse
import os
import glob

import numpy as np
import trimesh
from pysdf import SDF


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing OBJ meshes"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder where NPZ files will be saved"
    )

    parser.add_argument(
        "--num_points",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--surface_sigma",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    return parser.parse_args()


def normalize_mesh(vertices):
    center = vertices.mean(axis=0)
    vertices = vertices - center

    scale = np.abs(vertices).max()
    vertices = vertices / scale

    return vertices.astype(np.float32), center, scale


def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    mesh_files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.obj"))
    )

    print(f"Found {len(mesh_files)} meshes")

    for mesh_file in mesh_files:

        print("\n==============================")
        print("Processing:", mesh_file)

        mesh = trimesh.load(mesh_file, force="mesh")

        print("Watertight:", mesh.is_watertight)
        print("Vertices:", len(mesh.vertices))
        print("Faces:", len(mesh.faces))

        verts = np.asarray(mesh.vertices)

        verts, center, scale = normalize_mesh(verts)

        mesh.vertices = verts

        faces = np.asarray(mesh.faces)

        sdf_fn = SDF(
            verts.astype(np.float64),
            faces.astype(np.int32)
        )

        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)

        extent = bbox_max - bbox_min

        bbox_min -= args.padding * extent
        bbox_max += args.padding * extent

        # --------------------------------------------------
        # 50% uniform points
        # --------------------------------------------------

        n_uniform = args.num_points // 2

        uniform_points = rng.uniform(
            low=bbox_min,
            high=bbox_max,
            size=(n_uniform, 3)
        )

        # --------------------------------------------------
        # 50% near-surface points
        # --------------------------------------------------

        n_surface = args.num_points - n_uniform

        surface_points = mesh.sample(n_surface)

        noise = rng.normal(
            loc=0.0,
            scale=args.surface_sigma,
            size=surface_points.shape
        )

        surface_points = surface_points + noise

        # --------------------------------------------------
        # Combine
        # --------------------------------------------------

        points = np.concatenate(
            [uniform_points, surface_points],
            axis=0
        ).astype(np.float32)

        sdf = -sdf_fn(points).astype(np.float32)

        print("SDF statistics:")
        print("  min:", sdf.min())
        print("  max:", sdf.max())
        print("  mean:", sdf.mean())
        print("  inside ratio:", np.mean(sdf < 0))

        name = os.path.splitext(
            os.path.basename(mesh_file)
        )[0]

        save_path = os.path.join(
            args.output_dir,
            f"{name}.npz"
        )

        np.savez_compressed(
            save_path,
            points=points,
            sdf=sdf,
            center=center.astype(np.float32),
            scale=np.float32(scale)
        )

        print("Saved:", save_path)

    print("\nDone.")


if __name__ == "__main__":
    main()