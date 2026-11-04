"""Camera geometry shared across stages."""
from __future__ import annotations

import numpy as np


def backproject(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray | None = None,
                mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lift a depth map to 3D points.

    Returns (points (N,3) in world frame if T_wc given else camera frame, pixel indices (N,2) as (v,u)).
    Pixels with depth <= 0 or outside `mask` are dropped.
    """
    h, w = depth.shape
    valid = depth > 0
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    if T_wc is not None:
        pts = pts @ T_wc[:3, :3].T + T_wc[:3, 3]
    return pts, np.stack([v, u], axis=1)


def project(points_w: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into a camera. Returns (uv (N,2) float, depth (N,))."""
    T_cw = np.linalg.inv(T_wc)
    pc = points_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def disparity_to_depth(disp: np.ndarray, fx: float, baseline_m: float, min_disp: float = 0.5) -> np.ndarray:
    """Rectified-stereo disparity (pixels) -> metric depth; invalid where disparity is tiny."""
    depth = np.zeros_like(disp, dtype=np.float32)
    ok = disp > min_disp
    depth[ok] = fx * baseline_m / disp[ok]
    return depth
