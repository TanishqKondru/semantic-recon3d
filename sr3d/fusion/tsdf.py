"""Volumetric TSDF fusion of posed depth maps (Curless & Levoy 1996) on Open3D's GPU voxel-block grid.

Instruments move independently of the anatomy, so their pixels are dropped before integration: the volume
is a static tissue map. Masks are dilated because stereo depth bleeds across instrument edges.

(Open3D 0.20's legacy ScalableTSDFVolume returns empty meshes on this setup; the tensor VoxelBlockGrid works.)
"""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c


def tissue_depth(depth: np.ndarray, instrument: np.ndarray | None, dilate_px: int = 5) -> np.ndarray:
    """Depth with instrument pixels (and a `dilate_px` border around them) set to 0."""
    if instrument is None or not instrument.any():
        return depth
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        instrument = cv2.dilate(instrument.astype(np.uint8), k) > 0
    return np.where(instrument, 0, depth).astype(np.float32)


class TSDFMap:
    def __init__(self, K: np.ndarray, voxel_m: float = 0.5e-3, trunc_voxels: float = 5.0,
                 max_depth_m: float = 0.2, block_count: int = 15000, device: str = "CUDA:0"):
        self.device = o3c.Device(device)
        self.K = o3c.Tensor(K)
        self.trunc, self.max_depth = trunc_voxels, max_depth_m
        self.grid = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=(1, 1, 3), voxel_size=voxel_m, block_resolution=16, block_count=block_count,
            device=self.device)

    def integrate(self, rgb: np.ndarray, depth: np.ndarray, T_wc: np.ndarray) -> None:
        """rgb uint8 (H, W, 3), depth metres (0 = skip), T_wc camera-to-world."""
        d = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth, np.float32))).to(self.device)
        c = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb, np.float32) / 255)).to(self.device)
        T_cw = o3c.Tensor(np.linalg.inv(T_wc))
        blocks = self.grid.compute_unique_block_coordinates(d, self.K, T_cw, 1.0, self.max_depth, self.trunc)
        self.grid.integrate(blocks, d, c, self.K, T_cw, 1.0, self.max_depth, self.trunc)

    def mesh(self, min_weight: float = 3.0) -> o3d.geometry.TriangleMesh:
        """Surface seen in at least `min_weight` frames, vertex-coloured, as a legacy mesh."""
        m = self.grid.extract_triangle_mesh(weight_threshold=min_weight).to_legacy()
        m.compute_vertex_normals()
        return m
