"""
Synthetic validation for MoGe2 panorama split/merge orientation and seam alignment.

This script does NOT require model weights. It simulates per-view predictions for a unit sphere centered at the panorama origin.
It then performs the same z-buffer world-merge used by the MoGe2Panorama node and reports alignment/metric errors.

Usage:
  - Ensure Python env with numpy, cv2, and utils3d installed.
  - Run: python scripts/validate_panorama.py

Outputs:
  - Prints mean absolute distance error (should be ~0 with correct rotation handling).
  - Writes debug images under ./outputs/validate_panorama/ (difference heatmap and mask).
"""
import os
import sys
import math
from pathlib import Path
import numpy as np
import cv2

# Ensure repo root is on sys.path so `moge` can be imported when running from anywhere
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import math

# Minimal geometry utils to avoid external deps

def image_uv(width: int, height: int, dtype=np.float32):
    u = np.linspace(0.0, 1.0, width, dtype=dtype)
    v = np.linspace(0.0, 1.0, height, dtype=dtype)
    uu, vv = np.meshgrid(u, v, indexing='xy')
    return np.stack([uu, vv], axis=-1)


def spherical_uv_to_directions(uv: np.ndarray):
    theta = (1.0 - uv[..., 0]) * (2.0 * np.pi)
    phi = uv[..., 1] * np.pi
    dirs = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)
    return dirs


def intrinsics_from_fov(fov_x_rad: float, fov_y_rad: float):
    fx = 0.5 / math.tan(fov_x_rad / 2.0)
    fy = 0.5 / math.tan(fov_y_rad / 2.0)
    K = np.array([[fx, 0.0, 0.5], [0.0, fy, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def project_cv(dirs_world: np.ndarray, R_wc: np.ndarray, K: np.ndarray):
    """Project world directions into a camera with rotation R (world->camera) and normalized intrinsics K.
    Returns uv in [0,1], depth=z_cam.
    """
    # Row-vector mapping: v_cam = v_world @ R^T
    v_cam = dirs_world @ R_wc.T
    z = v_cam[..., 2]
    x = v_cam[..., 0]
    y = v_cam[..., 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # Avoid division by zero; caller should mask z>0
    eps = 1e-8
    u = fx * (x / (z + eps)) + cx
    v = fy * (y / (z + eps)) + cy
    uv = np.stack([u, v], axis=-1)
    return uv, z


def unproject_cv(uv: np.ndarray, R_wc: np.ndarray, K: np.ndarray):
    """Convert per-camera pixel UV in [0,1] to world-space ray directions.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (uv[..., 0] - cx) / fx
    y_cam = (uv[..., 1] - cy) / fy
    z_cam = np.ones_like(x_cam, dtype=np.float32)
    v_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    v_cam = v_cam / (np.linalg.norm(v_cam, axis=-1, keepdims=True) + 1e-9)
    # Row-vector: v_world = v_cam @ R
    v_world = v_cam @ R_wc
    return v_world


def look_at_rotation(forward_world: np.ndarray, up_world: np.ndarray = np.array([0, 0, 1], dtype=np.float32)):
    f = forward_world.astype(np.float32)
    f = f / (np.linalg.norm(f) + 1e-9)
    up = up_world.astype(np.float32)
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0, 1, 0], dtype=np.float32)
    x_axis = np.cross(up, f)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    y_axis = np.cross(f, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    z_axis = f
    # Rows are camera axes in world space for world->camera mapping
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    return R.astype(np.float32)


def get_cube_cameras():
    forwards = [
        np.array([1, 0, 0], dtype=np.float32),
        np.array([-1, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0, -1, 0], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32),
        np.array([0, 0, -1], dtype=np.float32),
    ]
    R_list = [look_at_rotation(f) for f in forwards]
    K = intrinsics_from_fov(math.radians(90.0), math.radians(90.0))
    K_list = [K for _ in R_list]
    return R_list, K_list


def simulate_per_view_sphere(points_world_gt: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, face_resolution: int):
    """
    Simulate per-view camera-space point maps for a unit sphere at origin.
    points_world_gt: (H, W, 3) ground-truth world points for the panorama, taken along spherical directions with fixed radius.
    Returns per-view (points_cam_list, normals_cam_list, masks_list, view_images_dummy)
    """
    num_views = len(extrinsics)
    # Build each view's camera-grid rays in world, then rotate into camera
    uv = image_uv(width=face_resolution, height=face_resolution)
    points_list = []
    normals_list = []
    masks_list = []
    images_list = []
    for i in range(num_views):
        # Directions in world for each view pixel (H,W,3)
        # Build camera rays directly from pixel UV
        R = extrinsics[i].astype(np.float32)
        K = intrinsics[i].astype(np.float32)
        d_cam = np.stack([
            (uv[..., 0] - K[0, 2]) / K[0, 0],
            (uv[..., 1] - K[1, 2]) / K[1, 1],
            np.ones_like(uv[..., 0], dtype=np.float32)
        ], axis=-1)
        d_cam = d_cam / (np.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-9)
        # Unit sphere: point on surface is at radius=1 along the ray.
        pts_cam = d_cam.astype(np.float32)
        n_cam = d_cam.astype(np.float32)
        mask = np.ones((face_resolution, face_resolution), dtype=bool)
        img = ((np.clip(d_cam, -1, 1) * 0.5 + 0.5) * 255).astype(np.uint8)  # simple direction-color debug
        points_list.append(pts_cam)
        normals_list.append(n_cam)
        masks_list.append(mask)
        images_list.append(img)
    return points_list, normals_list, masks_list, images_list


def merge_zbuffer(pano_h: int, pano_w: int, per_points, per_normals, per_masks, extrinsics, intrinsics):
    # Prepare pano rays (world directions)
    uv = image_uv(width=pano_w, height=pano_h)
    spherical_dirs = spherical_uv_to_directions(uv)

    best_dist = np.full((pano_h, pano_w), np.inf, dtype=np.float32)
    best_point_world = np.full((pano_h, pano_w, 3), np.nan, dtype=np.float32)
    best_normal_world = np.full((pano_h, pano_w, 3), 0, dtype=np.float32)
    final_mask = np.zeros((pano_h, pano_w), dtype=bool)

    for i in range(len(per_points)):
        pts = per_points[i]
        msk = per_masks[i]
        if pts is None or msk is None:
            continue
        projected_uv, projected_depth = project_cv(spherical_dirs, R_wc=extrinsics[i], K=intrinsics[i])
        proj_valid = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
        proj_clamped = np.clip(projected_uv, 0, 1)
        mapx = (proj_clamped[..., 0] * (pts.shape[1] - 1)).astype(np.float32)
        mapy = (proj_clamped[..., 1] * (pts.shape[0] - 1)).astype(np.float32)
        remap_points = cv2.remap(pts, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        remap_mask = cv2.remap(msk.astype(np.uint8), mapx, mapy, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
        valid = proj_valid & remap_mask & np.isfinite(remap_points).all(axis=-1)
        if not np.any(valid):
            continue
        R = extrinsics[i].astype(np.float32)
        pts_cam = remap_points.reshape(-1, 3)
        # Convert to world with row-vector convention: x_world = x_cam @ R
        pts_world = (pts_cam @ R).reshape(pano_h, pano_w, 3)
        dist = np.linalg.norm(pts_world, axis=-1)
        better = (dist < best_dist) & valid
        best_dist[better] = dist[better]
        best_point_world[better] = pts_world[better]
        final_mask[better] = True

        normal = per_normals[i]
        if normal is not None:
            remap_n = cv2.remap(normal, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            n_cam = remap_n.reshape(-1, 3)
            n_world = (n_cam @ R).reshape(pano_h, pano_w, 3)
            n_world = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-9)
            best_normal_world[better] = n_world[better]

    return best_point_world, best_normal_world, final_mask


def main():
    out_dir = Path("outputs/validate_panorama")
    out_dir.mkdir(parents=True, exist_ok=True)

    pano_h, pano_w = 512, 1024
    # Ground-truth world points on unit sphere for each pano pixel
    uv = image_uv(width=pano_w, height=pano_h)
    dirs_world = spherical_uv_to_directions(uv)
    pts_world_gt = dirs_world.astype(np.float32)  # radius=1

    # Use a simple cube camera setup (6 views, 90° FOV)
    extrinsics, intrinsics_list = get_cube_cameras()
    face_res = 256
    per_points, per_normals, per_masks, per_imgs = simulate_per_view_sphere(pts_world_gt, extrinsics, intrinsics_list, face_res)
    merged_points, merged_normals, merged_mask = merge_zbuffer(pano_h, pano_w, per_points, per_normals, per_masks, extrinsics, intrinsics_list)

    # Compute error
    valid = merged_mask
    diff = np.linalg.norm(merged_points - pts_world_gt, axis=-1)
    mae = float(np.mean(diff[valid])) if np.any(valid) else float("nan")
    print(f"Mean absolute 3D error (world): {mae:.6f} (should be ~0)")

    # Save debug images
    # Difference heatmap
    disp = (diff / (diff[valid].max() + 1e-6) if np.any(valid) else diff).astype(np.float32)
    heat = (np.clip(disp, 0, 1) * 255).astype(np.uint8)
    heat_rgb = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_dir / "diff_heatmap.png"), heat_rgb)
    cv2.imwrite(str(out_dir / "mask.png"), (valid.astype(np.uint8) * 255))


if __name__ == "__main__":
    main()
