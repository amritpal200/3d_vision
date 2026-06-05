# tools/npz_to_obj_ball_pivoting.py

import argparse
import os
import numpy as np
import open3d as o3d


def reconstruct_mesh(npz_path, obj_path):
    data = np.load(npz_path)

    if "surface_points" not in data:
        print(f"[WARN] {npz_path}: no surface_points found")
        return

    points = data["surface_points"]

    if len(points) < 100:
        print(f"[WARN] {npz_path}: too few points")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Use stored normals if available
    if "surface_normals" in data:
        normals = data["surface_normals"]

        if len(normals) == len(points):
            pcd.normals = o3d.utility.Vector3dVector(normals)
        else:
            pcd.estimate_normals()
    else:
        pcd.estimate_normals()

    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass

    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=2.0
    )

    if len(pcd.points) < 100:
        print(f"[WARN] {npz_path}: too few points after filtering")
        return

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    radii = [
        avg_dist,
        avg_dist * 2.0,
        avg_dist * 4.0,
        avg_dist * 8.0,
    ]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()

    os.makedirs(os.path.dirname(obj_path), exist_ok=True)

    success = o3d.io.write_triangle_mesh(
        obj_path,
        mesh,
        write_ascii=True
    )

    if success:
        print(f"[OK] {obj_path}")
    else:
        print(f"[FAIL] {obj_path}")


def convert_split(dataroot, split):
    sdf_root = os.path.join(
        dataroot,
        "sdf",
        split
    )

    obj_root = os.path.join(
        dataroot,
        "obj",
        split
    )

    os.makedirs(obj_root, exist_ok=True)

    npz_files = sorted(
        f for f in os.listdir(sdf_root)
        if f.endswith(".npz")
    )

    print(f"Found {len(npz_files)} NPZ files")

    for idx, filename in enumerate(npz_files):
        npz_path = os.path.join(sdf_root, filename)

        obj_name = filename.replace(".npz", ".obj")
        obj_path = os.path.join(obj_root, obj_name)

        print(
            f"[{idx+1}/{len(npz_files)}] "
            f"{filename}"
        )

        reconstruct_mesh(
            npz_path,
            obj_path
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataroot",
        required=True
    )

    parser.add_argument(
        "--split",
        default="train_pairs"
    )

    args = parser.parse_args()

    convert_split(
        args.dataroot,
        args.split
    )


if __name__ == "__main__":
    main()