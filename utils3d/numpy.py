from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ArrayLike = Union[np.ndarray, Sequence[float]]
EPS = 1e-8

__all__ = [
    "depth_edge",
    "extrinsics_look_at",
    "icosahedron",
    "image_mesh",
    "image_pixel_center",
    "image_uv",
    "intrinsics_from_fov",
    "project_cv",
    "sliding_window_2d",
    "uv_to_pixel",
]


def image_uv(width: int, height: int, dtype=np.float32) -> np.ndarray:
    """Return normalized UV coordinates in [0, 1] for an image grid."""
    u = np.linspace(0.0, 1.0, width, dtype=dtype)
    v = np.linspace(0.0, 1.0, height, dtype=dtype)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu, vv], axis=-1)


def image_pixel_center(width: int, height: int, dtype=np.float32) -> np.ndarray:
    """Return pixel-centre coordinates measured in pixels."""
    x = np.linspace(0.5, width - 0.5, width, dtype=dtype)
    y = np.linspace(0.5, height - 0.5, height, dtype=dtype)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack([xx, yy], axis=-1)


def sliding_window_2d(arr: np.ndarray, window_shape: Tuple[int, int], step: int = 1, axis: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """Lightweight sliding window helper matching utils3d.numpy API."""
    if isinstance(window_shape, int):
        window_shape = (window_shape, window_shape)
    if isinstance(step, int):
        step = (step, step)

    axis = tuple(a if a >= 0 else arr.ndim + a for a in axis)
    if len(axis) != 2:
        raise ValueError("Only 2D sliding windows are supported in this stub.")

    view = sliding_window_view(arr, window_shape, axis=axis)
    # sampling stride handling is limited to gaps along the first two axes
    return view[::step[0], ::step[1], ...]


def depth_edge(depth: np.ndarray, rtol: float = 0.04, mask: np.ndarray | None = None) -> np.ndarray:
    """Detect large relative jumps in the depth map and mark them as edges."""
    depth = depth.astype(np.float32)
    if mask is None:
        mask = np.isfinite(depth)
    else:
        mask = mask.astype(bool)

    edges = np.zeros_like(depth, dtype=bool)
    for axis in (0, 1):
        diff = np.abs(np.diff(depth, axis=axis))
        if axis == 0:
            valid = mask[:-1, ...] & mask[1:, ...]
            baseline = np.maximum(np.abs(depth[:-1, ...]), np.abs(depth[1:, ...]))
            cond = valid & (diff > (baseline * rtol + 1e-6))
            edges[:-1, ...] |= cond
            edges[1:, ...] |= cond
        else:
            valid = mask[..., :-1] & mask[..., 1:]
            baseline = np.maximum(np.abs(depth[..., :-1]), np.abs(depth[..., 1:]))
            cond = valid & (diff > (baseline * rtol + 1e-6))
            edges[..., :-1] |= cond
            edges[..., 1:] |= cond
    return edges


def icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    """Return vertices and triangular faces of a unit icosahedron."""
    phi = (1 + 5 ** 0.5) / 2
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            verts.append([sx, sy * phi, 0.0])
    for sy in (-1, 1):
        for sz in (-1, 1):
            verts.append([0.0, sy, sz * phi])
    for sx in (-1, 1):
        for sz in (-1, 1):
            verts.append([sx * phi, 0.0, sz])
    vertices = np.asarray(verts, dtype=np.float32)
    vertices /= np.linalg.norm(vertices, axis=-1, keepdims=True)

    faces = np.asarray([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)
    return vertices, faces


def intrinsics_from_fov(fov_x: float, fov_y: float) -> np.ndarray:
    fx = 0.5 / math.tan(fov_x / 2)
    fy = 0.5 / math.tan(fov_y / 2)
    return np.array(
        [[fx, 0.0, 0.5], [0.0, fy, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def extrinsics_look_at(origin: ArrayLike, target: ArrayLike, up: ArrayLike) -> np.ndarray:
    origins = np.asarray(origin, dtype=np.float32)
    targets = np.asarray(target, dtype=np.float32)
    ups = np.asarray(up, dtype=np.float32)

    if targets.ndim == 1:
        targets = targets[None, :]
    if origins.ndim == 1:
        origins = np.broadcast_to(origins, targets.shape)
    if ups.ndim == 1:
        ups = np.broadcast_to(ups, targets.shape)

    extrinsics = []
    for o, t, up_vec in zip(origins, targets, ups):
        forward = _normalize(t - o)
        up_dir = _normalize(up_vec)
        if abs(np.dot(forward, up_dir)) > 0.999:
            up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = _normalize(np.cross(up_dir, forward))
        up_ortho = _normalize(np.cross(forward, right))
        R = np.stack([right, up_ortho, forward], axis=0)
        t_vec = -(R @ o.astype(np.float32))
        extrinsics.append(np.concatenate([R, t_vec[:, None]], axis=1))
    return np.stack(extrinsics, axis=0)


def project_cv(points: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = points.astype(np.float32)
    extrinsics = extrinsics.astype(np.float32)
    if extrinsics.shape == (4, 4):
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
    elif extrinsics.shape == (3, 4):
        R = extrinsics[:, :3]
        t = extrinsics[:, 3]
    elif extrinsics.shape == (3, 3):
        R = extrinsics
        t = np.zeros(3, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected extrinsic shape: {extrinsics.shape}")

    intrinsics = intrinsics.astype(np.float32)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    pts_cam = points @ R.T + t
    z = pts_cam[..., 2]
    x = pts_cam[..., 0]
    y = pts_cam[..., 1]
    u = fx * (x / (z + EPS)) + cx
    v = fy * (y / (z + EPS)) + cy
    uv = np.stack([u, v], axis=-1)
    return uv, z


def uv_to_pixel(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    mapx = (np.clip(uv[..., 0], 0.0, 1.0) * (width - 1)).astype(np.float32)
    mapy = (np.clip(uv[..., 1], 0.0, 1.0) * (height - 1)).astype(np.float32)
    return np.stack([mapx, mapy], axis=-1)


def _build_faces(width: int, height: int, index_map: np.ndarray, tri: bool) -> np.ndarray:
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            ids = [y * width + x, y * width + x + 1, (y + 1) * width + x, (y + 1) * width + x + 1]
            remapped = index_map[ids]
            if np.any(remapped < 0):
                continue
            a, b, c, d = remapped
            if tri:
                faces.append([a, c, b])
                faces.append([c, d, b])
            else:
                faces.append([a, b, d, c])
    return np.asarray(faces, dtype=np.int32)


def image_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    uvs: np.ndarray,
    normals: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    tri: bool = True,
):
    """Convert per-pixel predictions into a triangle mesh."""
    height, width = points.shape[:2]
    mask_flat = np.ones(height * width, dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
    if not np.any(mask_flat):
        return (
            np.zeros((0, 3 if tri else 4), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, colors.shape[-1]), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        ) if normals is None else (
            np.zeros((0, 3 if tri else 4), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, colors.shape[-1]), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    vertices = points.reshape(-1, 3).astype(np.float32)
    vertex_colors = colors.reshape(-1, colors.shape[-1]).astype(np.float32)
    vertex_uvs = uvs.reshape(-1, 2).astype(np.float32)
    vertex_normals = normals.reshape(-1, 3).astype(np.float32) if normals is not None else None

    valid_idx = np.where(mask_flat)[0]
    index_map = np.full(height * width, -1, dtype=np.int32)
    index_map[valid_idx] = np.arange(len(valid_idx), dtype=np.int32)

    vertices = vertices[valid_idx]
    vertex_colors = vertex_colors[valid_idx]
    vertex_uvs = vertex_uvs[valid_idx]
    if vertex_normals is not None:
        vertex_normals = vertex_normals[valid_idx]

    faces = _build_faces(width, height, index_map, tri)

    if normals is None:
        return faces, vertices, vertex_colors, vertex_uvs
    else:
        return faces, vertices, vertex_colors, vertex_uvs, vertex_normals
