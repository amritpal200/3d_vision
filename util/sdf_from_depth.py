import os
import numpy as np
from scipy.spatial import cKDTree


def backproject_ortho_depth(depth, xmag=1.0, ymag=1.0, cam_pose=None):
    """
    Backproject an orthographic depth map into world coordinates.

    depth: HxW numpy array (depth along camera z)
    xmag, ymag: orthographic magnification (half-width/half-height in world units)
    cam_pose: 4x4 camera-to-world matrix. If provided, points are returned in world coords.

    Returns: pts (M,3) array of 3D points (world coords if cam_pose provided),
             mask_info (mask,H,W) where mask is boolean mask of valid depth>0
    """
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    Z = depth.reshape(-1)
    mask = Z > 0
    if not np.any(mask):
        return np.zeros((0,3), dtype=np.float32), (mask.reshape(H, W), (H, W))

    uu = uu.reshape(-1)[mask].astype(np.float32)
    vv = vv.reshape(-1)[mask].astype(np.float32)
    Z = Z[mask].astype(np.float32)

    # map pixel coords to camera x,y in orthographic projection
    # u in [0,W-1] -> x in [-xmag, xmag]
    x = (uu / (W - 1) * 2.0 - 1.0) * xmag
    # v in [0,H-1] -> y in [ymag, -ymag] (flip y)
    y = (1.0 - vv / (H - 1) * 2.0) * ymag

    pts_cam = np.stack([x, y, Z], axis=1)
    if cam_pose is not None:
        # cam_pose is camera->world; convert pts to homogeneous and transform
        ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([pts_cam, ones], axis=1)
        pts_world = (cam_pose @ homo.T).T[:, :3]
        return pts_world, (mask.reshape(H, W), (H, W))
    else:
        return pts_cam, (mask.reshape(H, W), (H, W))


def sample_points_near_surface(surface_pts, N, sigma=0.01):
    """Sample N query points by jittering surface points with gaussian noise."""
    if surface_pts.shape[0] == 0:
        return np.random.uniform(-0.5, 0.5, size=(N, 3)).astype(np.float32)
    idx = np.random.randint(0, surface_pts.shape[0], size=N)
    pts = surface_pts[idx] + np.random.normal(scale=sigma, size=(N, 3)).astype(np.float32)
    return pts


def compute_signed_sdf(query_pts, surface_pts, depth_map=None, cam_pose=None, xmag=1.0, ymag=1.0):
    """
    Compute signed distances for query_pts given a surface point cloud and optional depth map/camera.

    query_pts: (N,3) in world coords
    surface_pts: (M,3) in world coords
    depth_map: optional HxW depth in camera coords (same as used to create surface_pts)
    cam_pose: camera->world matrix used to create depth_map

    Returns: sdf (N,) signed distances
    """
    if surface_pts.shape[0] == 0:
        return np.ones(len(query_pts), dtype=np.float32) * 1e3

    tree = cKDTree(surface_pts)
    # scipy versions differ in supported kwargs; avoid `n_jobs` for compatibility
    dists, _ = tree.query(query_pts, k=1)

    signs = np.ones(len(query_pts), dtype=np.float32)
    if depth_map is not None and cam_pose is not None:
        # project query points into camera coords to compare z
        world_to_cam = np.linalg.inv(cam_pose)
        ones = np.ones((query_pts.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([query_pts, ones], axis=1)
        pts_cam = (world_to_cam @ homo.T).T[:, :3]
        X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        H, W = depth_map.shape
        # pixel coords from orthographic mapping
        u = ((X / xmag) + 1.0) * 0.5 * (W - 1)
        v = (1.0 - (Y / ymag) * 0.5 * (H - 1))  # approximate; correct below
        # correct v mapping
        v = (1.0 - (Y / ymag)) * 0.5 * (H - 1)

        for i in range(len(query_pts)):
            ui = int(round(u[i]))
            vi = int(round(v[i]))
            if ui < 0 or ui >= W or vi < 0 or vi >= H:
                continue
            d_map = float(depth_map[vi, ui])
            tol = 1e-3
            if Z[i] > d_map + tol:
                signs[i] = 1.0
            else:
                signs[i] = -1.0

    sdf = dists * signs
    return sdf
