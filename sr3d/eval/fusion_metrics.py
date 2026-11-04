"""Score a fused map on held-out frames: render it from their camera poses and compare with what they saw.

There is no ground-truth geometry, so the rendered depth is compared with the held-out frame's own stereo
depth, and the rendered colour with its image, on tissue pixels only (instruments are not in the map).
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


class MeshRenderer:
    """Ray-casts a vertex-coloured triangle mesh into pinhole cameras."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.tris = np.asarray(mesh.triangles)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                                 o3d.core.Tensor(self.tris.astype(np.uint32)))
        self.colors = np.asarray(mesh.vertex_colors)

    def render(self, K: np.ndarray, T_wc: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Depth (H, W) metres along the optical axis, 0 = no hit; RGB (H, W, 3) in [0, 255]."""
        v, u = np.mgrid[0:height, 0:width]
        # ray direction with unit z in the camera frame, so the hit distance t is the depth
        d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
        d_w = d_cam.reshape(-1, 3) @ T_wc[:3, :3].T
        o_w = np.broadcast_to(T_wc[:3, 3], d_w.shape)
        rays = o3d.core.Tensor(np.hstack([o_w, d_w]).astype(np.float32))
        hit = self.scene.cast_rays(rays)
        t = hit["t_hit"].numpy()
        ok = np.isfinite(t)
        depth = np.where(ok, t, 0).reshape(height, width).astype(np.float32)
        rgb = np.zeros((height * width, 3))
        if self.colors.size:
            prim = hit["primitive_ids"].numpy()[ok]
            bu, bv = hit["primitive_uvs"].numpy()[ok].T
            tri = self.tris[prim]
            rgb[ok] = ((1 - bu - bv)[:, None] * self.colors[tri[:, 0]] + bu[:, None] * self.colors[tri[:, 1]]
                       + bv[:, None] * self.colors[tri[:, 2]]) * 255
        return depth, rgb.reshape(height, width, 3)


def heldout_scores(rendered_depth, rendered_rgb, depth, rgb, tissue) -> dict:
    """depth_mm: median |rendered - stereo| (mm); coverage: share of tissue pixels the map covers;
    psnr: rendered vs observed colour (dB) on covered tissue pixels."""
    covered = tissue & (rendered_depth > 0)
    both = covered & (depth > 0)
    mse = np.mean((rendered_rgb[covered] - rgb[covered].astype(float)) ** 2) if covered.any() else np.nan
    return {
        "depth_mm": 1000 * float(np.median(np.abs(rendered_depth - depth)[both])) if both.any() else np.nan,
        "coverage": float(covered.sum() / max(tissue.sum(), 1)),
        "psnr": float(10 * np.log10(255 ** 2 / mse)) if covered.any() else np.nan,
    }
