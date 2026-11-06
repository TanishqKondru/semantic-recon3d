"""Shape prior for the instrument: recover the surface the camera cannot see.

Stereo only sees the camera-facing half of an instrument shaft, but a hovering shaft is closest to the tissue
on its underside. Measured on synthetic scenes with known geometry (docs/ROS.md), the visible-surface distance
reads a hovering 8 mm shaft 2.6 mm too far before any stereo error, and the 2 mm alarm misses it.

The robot knows which instrument is mounted, and da Vinci Xi shafts are cylinders of a catalogue diameter
(8 mm). This module fits a capsule (cylinder + hemispherical tip) of that known radius to the visible
instrument points and measures the distance from the whole capsule surface to the tissue.

* Axis: robust least squares (soft L1) of |distance to axis| - r, initialised from PCA (direction) and the
  centroid shifted by 2r/pi away from the camera (the centroid of a half circle seen from one side).
* Tip: the centre of the hemisphere, fitted along the axis to the points near the far end; when the tip is
  out of view the capsule simply ends at the last visible point.
* Gate (fixed a priori, not tuned): at least `min_points` points, visible length >= 3 r along the axis, RMS
  residual <= `max_rms`. Otherwise no model distance is returned.

The caller reports min(visible distance, model distance): a fitted surface can only make the system more
cautious. Real instrument tips (jaws, wrist) are thinner than the shaft, so the capsule over-covers them,
which errs in the same, cautious, direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class CapsuleFit:
    centre: np.ndarray      # (3,) point on the axis
    axis: np.ndarray        # (3,) unit, pointing towards the tip
    s_start: float          # axis coordinate of the shaft end of the visible part
    s_tip: float            # axis coordinate of the hemisphere centre
    radius: float
    rms: float              # RMS of |distance to axis| - r over the inliers, metres (NaN for "offset")
    n_points: int
    kind: str = "fit"       # "fit": cylinder fitted to the surface; "offset": axis placed one radius behind it

    def point(self, s: float) -> np.ndarray:
        return self.centre + s * self.axis

    def surface_samples(self, step: float = 0.25e-3) -> np.ndarray:
        """Points on the cylinder between s_start and s_tip plus the tip hemisphere (world)."""
        a = self.axis
        x = np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(a, x)
        r = self.radius
        phi = np.linspace(0, 2 * np.pi, max(int(2 * np.pi * r / step), 8), endpoint=False)
        s = np.arange(self.s_start, self.s_tip + 1e-9, step)
        P, S = np.meshgrid(phi, s)
        ring = np.cos(P).reshape(-1, 1) * x + np.sin(P).reshape(-1, 1) * y
        shaft = self.centre + S.reshape(-1, 1) * a + r * ring
        th = np.linspace(0, np.pi / 2, max(int(np.pi / 2 * r / step), 4))
        T, P2 = np.meshgrid(th, phi)
        d = (np.cos(T).reshape(-1, 1) * a
             + np.sin(T).reshape(-1, 1) * (np.cos(P2).reshape(-1, 1) * x + np.sin(P2).reshape(-1, 1) * y))
        return np.concatenate([shaft, self.point(self.s_tip) + r * d])


def _axis_residual(params, pts, r):
    c, a = params[:3], params[3:]
    a = a / np.linalg.norm(a)
    v = pts - c
    radial = np.linalg.norm(v - (v @ a)[:, None] * a, axis=1)
    return radial - r


def fit_capsule(pts: np.ndarray, radius: float, cam_centre: np.ndarray, tip: np.ndarray | None = None,
                min_points: int = 200, max_rms: float = 1e-3, max_points: int = 1000,
                seed: int = 0) -> CapsuleFit | None:
    """Fit a capsule of known radius to visible instrument points (world). `tip` (3D tip estimate, or None
    when the tip is out of view) only decides which end of the axis is the tip."""
    if len(pts) < min_points:
        return None
    if len(pts) > max_points:     # the fit needs shape, not every pixel
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    m = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - m, full_matrices=False)
    a0 = vt[0]
    w = m - cam_centre
    w = w - (w @ a0) * a0
    if np.linalg.norm(w) < 1e-9:
        return None
    c0 = m + 2 * radius / np.pi * w / np.linalg.norm(w)
    sol = least_squares(_axis_residual, np.concatenate([c0, a0]), args=(pts, radius), loss="soft_l1",
                        f_scale=0.5e-3, max_nfev=100)
    c, a = sol.x[:3], sol.x[3:] / np.linalg.norm(sol.x[3:])
    res = _axis_residual(sol.x, pts, radius)
    inl = np.abs(res) < 3 * max_rms
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else np.inf
    s = (pts - c) @ a
    if tip is not None and (tip - m) @ a < 0:
        a, s = -a, -s
    s_lo, s_hi = float(np.quantile(s, 0.01)), float(np.quantile(s, 0.99))
    if rms > max_rms or s_hi - s_lo < 3 * radius:
        return None
    s_tip = s_hi
    if tip is not None:
        # hemisphere centre along the axis: points near the far end lie at distance r from it
        near = pts[s > s_hi - 2 * radius]
        s_near = s[s > s_hi - 2 * radius]
        if len(near) >= 20:
            # only points beyond the centre are on the hemisphere; those before it are shaft (already fitted)
            f = lambda t: np.where(s_near > t[0], np.linalg.norm(near - (c + t[0] * a), axis=1) - radius, 0.0)
            t = least_squares(f, [s_hi - radius], loss="soft_l1", f_scale=0.5e-3,
                              bounds=([s_hi - 2 * radius], [s_hi])).x[0]
            s_tip = float(t)
    return CapsuleFit(c, a, s_lo, max(s_tip, s_lo), radius, rms, len(pts))


def offset_capsule(pts: np.ndarray, radius: float, cam_centre: np.ndarray, tip: np.ndarray | None = None,
                   min_points: int = 200) -> CapsuleFit | None:
    """Capsule whose axis is the visible points' principal line moved one radius away from the camera (no
    curvature needed). The tip hemisphere is centred on the last visible point, so it extends one radius
    beyond what is seen (cautious)."""
    if len(pts) < min_points:
        return None
    m = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - m, full_matrices=False)
    a = vt[0]
    if tip is not None and (tip - m) @ a < 0:
        a = -a
    w = m - cam_centre
    w = w - (w @ a) * a
    if np.linalg.norm(w) < 1e-9:
        return None
    c = m + radius * w / np.linalg.norm(w)
    s = (pts - c) @ a
    s_lo, s_hi = float(np.quantile(s, 0.01)), float(np.quantile(s, 0.99))
    if s_hi - s_lo < 3 * radius:
        return None
    return CapsuleFit(c, a, s_lo, s_hi, radius, np.nan, len(pts), kind="offset")


def instrument_capsule(pts, radius, cam_centre, tip=None, mode: str = "tiered") -> CapsuleFit | None:
    """mode "fit": curvature fit only; "offset": offset model only; "tiered": fit, else offset."""
    cap = fit_capsule(pts, radius, cam_centre, tip) if mode in ("fit", "tiered") else None
    if cap is None and mode in ("offset", "tiered"):
        cap = offset_capsule(pts, radius, cam_centre, tip)
    return cap
