"""Synthetic stereo scenes with known instrument-tissue distance, for absolute end-to-end accuracy.

StereoMIS has no measured tool-tissue distance, and the Stage 6 reference shares the stereo depth with the
system. Here the geometry is known exactly:

* Tissue: a real instrument-free StereoMIS frame lifted to a mesh with its own stereo depth (holes filled,
  Gaussian smoothed) and textured with the real image. The mesh *is* the ground truth surface.
* Instrument: a capsule (cylinder + hemispherical tip) of da Vinci size (8 mm diameter), with a
  surface-attached texture and headlight shading (Lambert + specular), entering from the image border.
  It is placed so that its exact minimum distance to the tissue mesh equals a chosen gap.
* Both rectified views are ray-cast (Open3D RaycastingScene) at the StereoMIS calibration, then blurred and
  given sensor noise, and fed to the real pipeline (RAFT -> mask -> TSDF map -> proximity).

Two reference distances: `true_gap` (whole instrument surface, what a safety monitor needs) and
`visible_gap` (only instrument points the left camera sees, the best a stereo system can observe).

Known simplifications: static scene and camera, no smoke / blood / motion blur, Lambertian tissue (lighting
baked into the texture), a synthetic instrument appearance (the segmenter was trained on real ones).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

TOOL, TISSUE = 1, 0


def tissue_mesh(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, smooth_px: float = 2.0, max_jump: float = np.inf):
    """Depth (m) + image -> (vertices (N, 3), triangles (M, 3), vertex colours (N, 3) float 0..1), camera frame.

    By default every grid triangle is kept, so the surface is closed (steep depth steps become steep walls).
    Dropping triangles across depth jumps (`max_jump`) leaves holes that render as black pixels."""
    d = depth.astype(np.float32).copy()
    hole = (d <= 0).astype(np.uint8)
    if hole.any():
        scale = d.max()
        d = cv2.inpaint((d / scale * 65535).astype(np.uint16), hole, 5, cv2.INPAINT_NS).astype(np.float32) / 65535 * scale
    d = cv2.GaussianBlur(d, (0, 0), smooth_px)
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # vertices sit a fraction of a pixel off the pixel grid: a ray through a vertex or along a shared edge can
    # slip between triangles (black speckles); the mesh, wherever its vertices are, is the ground truth
    uo, vo = u + 0.37, v + 0.29
    verts = np.stack([(uo - K[0, 2]) * d / K[0, 0], (vo - K[1, 2]) * d / K[1, 1], d], -1).reshape(-1, 3)
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tris = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, e], -1).reshape(-1, 3)])
    z = verts[:, 2][tris]
    tris = tris[(z.max(1) - z.min(1)) < max_jump]
    return verts, tris, rgb.reshape(-1, 3).astype(np.float32) / 255


@dataclass
class Capsule:
    tip_centre: np.ndarray    # (3,) centre of the hemispherical tip
    axis: np.ndarray          # (3,) unit vector from the tip towards the shaft
    radius: float = 4e-3
    length: float = 0.15

    def mesh(self, n_around: int = 96, n_along: int = 600) -> o3d.geometry.TriangleMesh:
        # cylinder along local z from 0 to `length` + sphere at the origin = rounded tip (shaft end out of view)
        cyl = o3d.geometry.TriangleMesh.create_cylinder(self.radius, self.length, n_around, n_along // 4)
        cyl.translate((0, 0, self.length / 2))
        m = cyl + o3d.geometry.TriangleMesh.create_sphere(self.radius, n_around // 2)
        # local z is the axis, tip at the origin -> world
        z = self.axis / np.linalg.norm(self.axis)
        x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z], 1)
        m.rotate(R, center=(0, 0, 0))
        m.translate(self.tip_centre)
        return m

    def surface_samples(self, step: float = 0.1e-3, reach: float = 0.03) -> np.ndarray:
        """Dense points on the tip hemisphere and the first `reach` of the shaft (world)."""
        z = self.axis / np.linalg.norm(self.axis)
        x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        n_phi = int(2 * np.pi * self.radius / step)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        s = np.arange(0, reach, step)
        P, S = np.meshgrid(phi, s)
        shaft = (self.tip_centre + S.reshape(-1, 1) * z
                 + self.radius * (np.cos(P).reshape(-1, 1) * x + np.sin(P).reshape(-1, 1) * y))
        th = np.linspace(0, np.pi / 2, int(np.pi / 2 * self.radius / step))
        T, P2 = np.meshgrid(th, phi)
        dirs = (np.cos(T).reshape(-1, 1) * -z
                + np.sin(T).reshape(-1, 1) * (np.cos(P2).reshape(-1, 1) * x + np.sin(P2).reshape(-1, 1) * y))
        return np.concatenate([shaft, self.tip_centre + self.radius * dirs])


class Scene:
    """Tissue mesh + optional capsule; renders rectified stereo pairs and answers exact distances."""

    def __init__(self, verts, tris, colours, K, baseline_m, size: tuple[int, int], seed: int = 0):
        """size = (height, width) of the rendered images."""
        self.K, self.B = K, baseline_m
        self.h, self.w = size
        self.tissue = o3d.t.geometry.TriangleMesh()
        self.tissue.vertex.positions = o3d.core.Tensor(verts.astype(np.float32))
        self.tissue.triangle.indices = o3d.core.Tensor(tris.astype(np.int32))
        self.tris, self.colours = tris, colours
        self.tissue_scene = o3d.t.geometry.RaycastingScene()
        self.tissue_scene.add_triangles(self.tissue)
        self.rng = np.random.default_rng(seed)
        tex = self.rng.normal(0, 1, (64, 512)).astype(np.float32)
        self.tool_tex = cv2.GaussianBlur(tex, (0, 0), 1.2)          # (angle, axial) texture, ~0.5 mm features
        self.tool_tex /= self.tool_tex.std()

    def tissue_distance(self, pts: np.ndarray) -> np.ndarray:
        return self.tissue_scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()

    def place(self, tissue_point: np.ndarray, normal: np.ndarray, axis: np.ndarray, gap: float,
              radius: float = 4e-3, iters: int = 6) -> Capsule:
        """Capsule whose exact minimum distance to the tissue equals `gap` (to ~1 um)."""
        cap = Capsule(tissue_point + normal * (radius + gap), axis / np.linalg.norm(axis), radius)
        for _ in range(iters):
            d = self.tissue_distance(cap.surface_samples()).min()
            cap.tip_centre = cap.tip_centre + normal * (gap - d)
        return cap

    def render(self, cap: Capsule | None, noise: float = 2.0, blur: float = 0.7):
        """-> left RGB, right RGB (uint8), left tool mask, left hit points (H, W, 3) and hit geometry ids."""
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(self.tissue)
        if cap is not None:
            tm = o3d.t.geometry.TriangleMesh.from_legacy(cap.mesh())
            scene.add_triangles(tm)
        out = []
        for cx in (0.0, self.B):      # left camera at the origin, right camera at +B along x (rectified)
            u, v = np.meshgrid(np.arange(self.w) + 0.0, np.arange(self.h) + 0.0)
            dirs = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1)
            dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
            orig = np.broadcast_to([cx, 0, 0], dirs.shape)
            rays = o3d.core.Tensor(np.concatenate([orig, dirs], -1).reshape(-1, 6).astype(np.float32))
            r = scene.cast_rays(rays)
            t = r["t_hit"].numpy()
            gid = r["geometry_ids"].numpy().astype(np.int64)
            prim = r["primitive_ids"].numpy().astype(np.int64)
            bary = r["primitive_uvs"].numpy()
            nrm = r["primitive_normals"].numpy()
            hit = np.isfinite(t)
            pts = orig.reshape(-1, 3) + dirs.reshape(-1, 3) * np.where(hit, t, 0)[:, None]
            img = np.zeros((len(t), 3), np.float32)
            ts = hit & (gid == TISSUE)
            tri = self.tris[prim[ts]]
            w1, w2 = bary[ts, 0:1], bary[ts, 1:2]
            img[ts] = (1 - w1 - w2) * self.colours[tri[:, 0]] + w1 * self.colours[tri[:, 1]] + w2 * self.colours[tri[:, 2]]
            tl = hit & (gid == TOOL)
            if tl.any():
                img[tl] = self._shade_tool(cap, pts[tl], nrm[tl], dirs.reshape(-1, 3)[tl])
            img = img.reshape(self.h, self.w, 3)
            if blur > 0:
                img = cv2.GaussianBlur(img, (0, 0), blur)
            img = img * 255 + self.rng.normal(0, noise, img.shape)
            out.append((np.clip(img, 0, 255).astype(np.uint8), (hit & (gid == TOOL)).reshape(self.h, self.w),
                        pts.reshape(self.h, self.w, 3), np.where(hit, gid, -1).reshape(self.h, self.w)))
        (left, mask, pts, gid), (right, *_) = out
        return left, right, mask, pts, gid

    def _shade_tool(self, cap: Capsule, p, n, d):
        z = cap.axis
        x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rel = p - cap.tip_centre
        s = rel @ z
        phi = np.arctan2(rel @ y, rel @ x)
        tu = ((phi + np.pi) / (2 * np.pi) * self.tool_tex.shape[0]).astype(int) % self.tool_tex.shape[0]
        tv = np.clip((s + cap.radius) / 0.5e-3, 0, self.tool_tex.shape[1] - 1).astype(int)
        albedo = 0.32 + 0.06 * self.tool_tex[tu, tv]
        albedo = np.where(s < 10e-3, albedo + 0.18, albedo)        # brighter metal jaw section near the tip
        lam = np.clip(-(n * d).sum(1), 0, 1)                         # headlight at the camera
        spec = lam ** 40 * 0.6
        g = np.clip(albedo * (0.2 + 0.8 * lam) + spec, 0, 1)
        return np.stack([g * 1.02, g, g * 0.98], 1)


def visible_gap(scene: Scene, pts: np.ndarray, gid: np.ndarray) -> float:
    """Exact minimum tissue distance over instrument surface points the left camera sees."""
    tool = gid == TOOL
    return float(scene.tissue_distance(pts[tool]).min()) if tool.any() else np.nan
