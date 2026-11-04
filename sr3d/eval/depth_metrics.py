"""Standard depth-estimation metrics (Eigen et al.) plus millimetre errors for surgical scenes."""
from __future__ import annotations

import numpy as np


def median_scale(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """Scale that aligns relative (mono) depth to metric ground truth."""
    return float(np.median(gt[valid]) / np.median(pred[valid]))


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None,
                  max_depth: float | None = None, align_scale: bool = False) -> dict:
    """Compare depth maps (metres). Only pixels where both are positive (and inside mask) count."""
    valid = (gt > 0) & (pred > 0) & np.isfinite(pred)
    if max_depth is not None:
        valid &= gt < max_depth
    if mask is not None:
        valid &= mask
    if not valid.any():
        return {"valid_frac": 0.0}
    s = median_scale(pred, gt, valid) if align_scale else 1.0
    p, g = pred[valid] * s, gt[valid]
    ratio = np.maximum(p / g, g / p)
    err = np.abs(p - g)
    return {
        "abs_rel": float(np.mean(err / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse_mm": float(np.sqrt(np.mean((p - g) ** 2)) * 1000),
        "mae_mm": float(np.mean(err) * 1000),
        "median_mm": float(np.median(err) * 1000),
        "delta1": float(np.mean(ratio < 1.25)),
        "valid_frac": float(valid.mean()),
        "scale": s,
    }


def aggregate(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
