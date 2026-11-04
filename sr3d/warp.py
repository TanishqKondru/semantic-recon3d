"""Depth- and pose-based view warping, used for consistency checks without ground-truth depth."""
from __future__ import annotations

import cv2
import numpy as np


def warp_to_reference(depth_i: np.ndarray, K: np.ndarray, T_wc_i: np.ndarray, T_wc_j: np.ndarray,
                      img_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise view i by sampling image j at where frame-i pixels land in camera j.

    Returns (warped image in frame i's pixel grid, validity mask).
    """
    h, w = depth_i.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth_i.astype(np.float64)
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1).reshape(-1, 3)
    T_ji = np.linalg.inv(T_wc_j) @ T_wc_i
    pj = pts @ T_ji[:3, :3].T + T_ji[:3, 3]
    zj = pj[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uj = (K[0, 0] * pj[:, 0] / zj + K[0, 2]).reshape(h, w).astype(np.float32)
        vj = (K[1, 1] * pj[:, 1] / zj + K[1, 2]).reshape(h, w).astype(np.float32)
    warped = cv2.remap(img_j, uj, vj, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (depth_i > 0) & (zj.reshape(h, w) > 0) & (uj >= 0) & (uj <= w - 1) & (vj >= 0) & (vj <= h - 1)
    return warped, valid


def photometric_error(img_i: np.ndarray, warped: np.ndarray, valid: np.ndarray) -> float:
    """Mean absolute grayscale error (0-255) over valid pixels."""
    a = cv2.cvtColor(img_i, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(a - b)[valid].mean()) if valid.any() else float("nan")
