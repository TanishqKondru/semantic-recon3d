"""Pose helpers: TUM trajectory files and SE(3) utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def to_matrix(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """(N,3) translations + (N,4) xyzw quaternions -> (N,4,4) transforms."""
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:, :3, 3] = t
    return T


def read_tum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read 'stamp tx ty tz qx qy qz qw' lines (comma or space separated) -> (stamps, (N,4,4))."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([float(v) for v in line.replace(",", " ").split()])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"{path}: expected 8 columns, got shape {arr.shape}")
    return arr[:, 0], to_matrix(arr[:, 1:4], arr[:, 4:8])


def write_tum(path: str | Path, stamps: np.ndarray, T: np.ndarray) -> None:
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw (camera-to-world, metres)\n")
        for s, t, qq in zip(stamps, T[:, :3, 3], q):
            f.write(f"{s:.6f} " + " ".join(f"{v:.9f}" for v in (*t, *qq)) + "\n")


def relative_to_first(T: np.ndarray) -> np.ndarray:
    """Re-express a trajectory so the first camera is the world origin."""
    return np.linalg.inv(T[0])[None] @ T
