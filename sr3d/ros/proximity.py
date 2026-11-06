"""Instrument-to-tissue proximity: the number a virtual fixture or a safety monitor would consume.

The live frame cannot see the tissue under an instrument (the tool occludes it), so the distance is
measured against the fused tissue map, which has seen that tissue from earlier viewpoints. Instrument
points come from the current frame's stereo depth inside the instrument mask, eroded away from both the mask
edge and holes in the depth (stereo depth bleeds across either). Distances are point-to-nearest-map-point
(k-d tree); with a 0.5 mm voxel map the discretisation error is below 0.25 mm. The reported minimum is taken
after a 5x5 median filter of the per-pixel distance image, so it needs a small patch of the instrument to be
that close, not one stray pixel.

Pure numpy / scipy: the ROS 2 node in `sr3d.ros.node` is a thin wrapper around this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from ..geometry import backproject


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k) > 0


@dataclass
class Tool:
    """One instrument (connected component of the mask) in the current frame."""
    pixels: int
    tip_px: tuple[int, int] | None    # (u, v) image position of the tip, None when it is out of view
    tip: np.ndarray | None            # (3,) world
    points: np.ndarray                # (N, 3) world, eroded-mask instrument surface
    distance: float = np.nan          # metres to the tissue map (robust minimum over `points`)
    closest_tool: np.ndarray | None = None    # (3,) instrument point that attains `distance`
    closest_tissue: np.ndarray | None = None  # (3,) its nearest map point
    distance_visible: float = np.nan  # from the visible instrument surface only
    distance_model: float = np.nan    # from the fitted capsule (`sr3d.ros.toolmodel`), NaN if not fitted
    capsule: object | None = None     # CapsuleFit when fitted


def find_tip(component: np.ndarray, margin: int = 5) -> tuple[int, int] | None:
    """Tip of an instrument seen as a mask component: the pixel farthest from where it enters the image.

    Laparoscopic instruments enter through the image border; the entry point is the centroid of the
    component's border pixels. A component that does not touch the border (shaft occluded) enters from the
    image edge nearest its centroid. Returns None when the farthest pixel lies within `margin` of the border,
    i.e. the tip is out of view.
    """
    v, u = np.nonzero(component)
    if len(u) == 0:
        return None
    h, w = component.shape
    on_border = (u == 0) | (u == w - 1) | (v == 0) | (v == h - 1)
    if on_border.any():
        eu, ev = u[on_border].mean(), v[on_border].mean()
    else:
        cu, cv = u.mean(), v.mean()
        eu, ev = min([(0, cv), (w - 1, cv), (cu, 0), (cu, h - 1)], key=lambda p: np.hypot(p[0] - cu, p[1] - cv))
    i = int(np.argmax((u - eu) ** 2 + (v - ev) ** 2))
    if min(u[i], v[i], w - 1 - u[i], h - 1 - v[i]) < margin:
        return None
    return int(u[i]), int(v[i])


def tip_point(depth: np.ndarray, mask: np.ndarray, uv: tuple[int, int], K: np.ndarray, T_wc: np.ndarray,
              window: int = 15) -> np.ndarray | None:
    """3D tip: the tip pixel lifted with the median valid instrument depth in a window around it."""
    u, v = uv
    h, w = depth.shape
    r = window // 2
    sl = np.s_[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)]
    d = depth[sl][mask[sl] & (depth[sl] > 0)]
    if len(d) == 0:
        return None
    z = float(np.median(d))
    p = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    return T_wc[:3, :3] @ p + T_wc[:3, 3]


class TissueModel:
    """Nearest-neighbour queries against one or more tissue point sets (world frame), e.g. the fused map
    plus the current frame's own tissue pixels. One k-d tree per set, so each can be rebuilt on its own."""

    def __init__(self, *point_sets: np.ndarray):
        self.sets = [np.asarray(p, np.float64) for p in point_sets if len(p)]
        self.trees = [cKDTree(p) for p in self.sets]

    @classmethod
    def merge(cls, *models: TissueModel | None) -> TissueModel:
        """Combine models without rebuilding their trees."""
        out = cls()
        for m in models:
            if m is not None:
                out.sets += m.sets
                out.trees += m.trees
        return out

    def __len__(self) -> int:
        return sum(len(p) for p in self.sets)

    def query(self, pts: np.ndarray, upper: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
        """-> (distance (N,), nearest tissue point (N, 3)); distances beyond `upper` come back as inf."""
        best_d = np.full(len(pts), np.inf)
        best_p = np.zeros((len(pts), 3))
        for p, tree in zip(self.sets, self.trees):
            d, j = tree.query(pts, k=1, distance_upper_bound=upper, workers=-1)
            closer = d < best_d
            best_d[closer], best_p[closer] = d[closer], p[j[closer]]
        return best_d, best_p


def measure(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, T_wc: np.ndarray, tissue: TissueModel | None,
            erode_px: int = 3, min_pixels: int = 200, median_px: int = 5,
            tool_radius: float | None = None, tool_mode: str = "tiered") -> list[Tool]:
    """Per-instrument tip and distance to tissue for one frame. depth in metres, mask bool, T_wc camera->world.

    With `tool_radius` (metres, from the instrument's specification), a capsule of that radius is fitted to the
    visible instrument points and the reported distance is min(visible-surface distance, capsule distance):
    the capsule covers the underside the camera cannot see (`sr3d.ros.toolmodel`)."""
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    core = erode(mask & (depth > 0), erode_px)
    tools = []
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_pixels:
            continue
        m = comp == c
        uv = find_tip(m)
        tip = None if uv is None else tip_point(depth, m & core, uv, K, T_wc)
        pts, vu = backproject(depth, K, T_wc, m & core)
        t = Tool(int(stats[c, cv2.CC_STAT_AREA]), uv, tip, pts)
        if tissue is not None and len(tissue) and len(pts):
            t.distance, t.closest_tool, t.closest_tissue = closest(pts, vu, tissue, median_px)
            t.distance_visible = t.distance
            if tool_radius:
                model_closest(t, tissue, tool_radius, T_wc[:3, 3], tool_mode)
        tools.append(t)
    return tools


def model_closest(t: Tool, tissue: TissueModel, radius: float, cam_centre: np.ndarray, mode: str = "tiered") -> None:
    """Distance from the instrument capsule to the tissue; updates `t` when it is closer than the visible surface.
    mode: "fit" (curvature fit), "offset" (axis one radius behind the visible points) or "tiered" (fit, else
    offset); see `sr3d.ros.toolmodel`."""
    from .toolmodel import instrument_capsule
    cap = instrument_capsule(t.points, radius, cam_centre, t.tip, mode)
    if cap is None:
        return
    t.capsule = cap
    samples = cap.surface_samples()
    upper = t.distance_visible + 1e-3 if np.isfinite(t.distance_visible) else np.inf
    d, nearest = tissue.query(samples, upper)
    i = int(np.argmin(d))
    t.distance_model = float(d[i]) if np.isfinite(d[i]) else np.inf
    if t.distance_model < t.distance:
        t.distance, t.closest_tool, t.closest_tissue = t.distance_model, samples[i], nearest[i]


def closest(pts: np.ndarray, vu: np.ndarray, tissue: TissueModel, median_px: int, margin_m: float = 2e-3):
    """Median-filtered minimum distance -> (distance, instrument point, tissue point).

    Far from the tissue every nearest-neighbour search visits thousands of map points, so a coarse pass over
    every 16th instrument point first bounds the answer. Distances beyond the bound cannot change a
    filtered minimum that lies below it, so the bounded result is exact; if it does not lie below the bound
    the query is repeated unbounded.
    """
    upper = tissue.query(pts[::16])[0].min() + margin_m
    for bound in (upper, np.inf):
        d, nearest = tissue.query(pts, bound)
        value, i = filtered_min(d, vu, median_px)
        if value < bound:
            return value, pts[i], nearest[i]
    return np.nan, None, None


def filtered_min(d: np.ndarray, vu: np.ndarray, size: int) -> tuple[float, int]:
    """Minimum of the `size` x `size` median-filtered distance image over the given pixels -> (value, index)."""
    if size <= 1:
        i = int(np.argmin(d))
        return float(d[i]), i
    lo = vu.min(0)
    shape = vu.max(0) - lo + 1
    img = np.full(shape, 1.0)   # 1 m stands for "not an instrument pixel": a mostly-empty window is never close
    img[vu[:, 0] - lo[0], vu[:, 1] - lo[1]] = d
    med = median_filter(img, size=size, mode="nearest")[vu[:, 0] - lo[0], vu[:, 1] - lo[1]]
    i = int(np.argmin(med))
    return float(med[i]), i


def min_distance(tools: list[Tool]) -> float:
    d = [t.distance for t in tools if np.isfinite(t.distance)]
    return min(d) if d else np.nan
