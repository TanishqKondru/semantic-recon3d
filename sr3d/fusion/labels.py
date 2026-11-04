"""Multi-view label voting: lift per-frame instrument probabilities onto 3D points.

Each point is projected into every frame; it counts as seen when the frame's own depth at that pixel agrees
with the point's depth (within `tol_m`), so points hidden behind an instrument or other tissue get no vote
from that frame. The point's instrument probability is the confidence-weighted mean of the votes it got.
`last` keeps only the most recent frame's vote, the single-frame baseline that voting is compared with.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..geometry import project


class Votes(NamedTuple):
    prob: np.ndarray  # (N,) weighted mean instrument probability, NaN where unseen
    last: np.ndarray  # (N,) probability from the last frame that saw the point, NaN where unseen
    seen: np.ndarray  # (N,) number of frames that saw the point


def vote(points: np.ndarray, frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]], K: np.ndarray,
         weights: list[float] | None = None, tol_m: float = 2e-3) -> Votes:
    """points (N, 3) world; frames = [(prob (H, W) in [0, 1], depth (H, W) metres, T_wc), ...] in time order."""
    n = len(points)
    num, den, seen = np.zeros(n), np.zeros(n), np.zeros(n, int)
    last = np.full(n, np.nan)
    for k, (prob, depth, T_wc) in enumerate(frames):
        w = 1.0 if weights is None else weights[k]
        h, wd = depth.shape
        uv, z = project(points, K, T_wc)
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inside = (z > 0) & (u >= 0) & (u < wd) & (v >= 0) & (v < h)
        idx = np.flatnonzero(inside)
        d = depth[v[idx], u[idx]]
        vis = idx[(d > 0) & (np.abs(d - z[idx]) < tol_m)]
        p = prob[v[vis], u[vis]]
        num[vis] += w * p
        den[vis] += w
        seen[vis] += 1
        last[vis] = p
    with np.errstate(invalid="ignore"):
        return Votes(num / den, last, seen)


def label_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    """IoU of the instrument class between two per-point probability arrays (threshold 0.5), seen points only."""
    ok = ~np.isnan(pred) & ~np.isnan(ref)
    p, r = pred[ok] > 0.5, ref[ok] > 0.5
    union = (p | r).sum()
    return float((p & r).sum() / union) if union else float("nan")
