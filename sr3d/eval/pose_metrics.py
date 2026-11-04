"""Trajectory metrics: ATE after Umeyama alignment and RPE (Sturm et al., IROS 2012)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares s, R, t such that dst ~= s * R @ src + t. src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_trajectory(T_est: np.ndarray, T_gt: np.ndarray, with_scale: bool = True) -> tuple[np.ndarray, float]:
    """Apply the Umeyama Sim(3)/SE(3) fit of camera centres to the whole estimated trajectory."""
    s, R, t = umeyama(T_est[:, :3, 3], T_gt[:, :3, 3], with_scale)
    out = T_est.copy()
    out[:, :3, :3] = R[None] @ T_est[:, :3, :3]
    out[:, :3, 3] = s * T_est[:, :3, 3] @ R.T + t
    return out, s


def ate_rmse(T_est: np.ndarray, T_gt: np.ndarray, with_scale: bool = True) -> float:
    aligned, _ = align_trajectory(T_est, T_gt, with_scale)
    return float(np.sqrt(np.mean(np.sum((aligned[:, :3, 3] - T_gt[:, :3, 3]) ** 2, axis=1))))


def rpe(T_est: np.ndarray, T_gt: np.ndarray, delta: int = 1, scale: float = 1.0) -> tuple[float, float]:
    """RMSE of relative translation (units of T_gt) and mean relative rotation error (deg) over `delta` steps."""
    te, re = [], []
    for i in range(len(T_gt) - delta):
        rel_gt = np.linalg.inv(T_gt[i]) @ T_gt[i + delta]
        rel_est = np.linalg.inv(T_est[i]) @ T_est[i + delta]
        rel_est[:3, 3] *= scale
        err = np.linalg.inv(rel_gt) @ rel_est
        te.append(np.linalg.norm(err[:3, 3]))
        re.append(np.degrees(Rotation.from_matrix(err[:3, :3]).magnitude()))
    return float(np.sqrt(np.mean(np.square(te)))), float(np.mean(re))


def evaluate(T_est: np.ndarray, T_gt: np.ndarray, delta: int = 1) -> dict:
    """ATE with Sim(3) alignment (scale-free methods), plus RPE using the recovered scale."""
    aligned, s = align_trajectory(T_est, T_gt, with_scale=True)
    ate = float(np.sqrt(np.mean(np.sum((aligned[:, :3, 3] - T_gt[:, :3, 3]) ** 2, axis=1))))
    rpe_t, rpe_r = rpe(T_est, T_gt, delta, scale=s)
    gt_len = float(np.linalg.norm(np.diff(T_gt[:, :3, 3], axis=0), axis=1).sum())
    return {"ate_rmse_mm": ate * 1000, "rpe_trans_mm": rpe_t * 1000, "rpe_rot_deg": rpe_r,
            "scale": s, "ate_pct_of_path": 100 * ate / max(gt_len, 1e-12)}
