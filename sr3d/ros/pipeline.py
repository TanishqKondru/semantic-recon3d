"""Online, causal perception loop behind the ROS 2 node: one stereo frame + camera pose in, proximity out.

Per frame: stereo depth -> instrument mask -> integrate tissue depth into the TSDF -> (every `map_every`
frames) refresh the k-d tree from the map's surface points -> per-instrument tip and distance. The map only
ever holds frames up to the current one, as it would on a robot, so tissue the camera has only just
uncovered is not in it yet; with `live_tissue` the current frame's own tissue pixels are queried as well.
Depth and mask functions are injected, so the offline evaluation (`scripts/eval_proximity.py`) can replay
cached outputs through exactly this code.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..fusion.tsdf import TSDFMap, tissue_depth
from ..geometry import backproject
from .proximity import TissueModel, Tool, measure, min_distance

DepthFn = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (left, right) RGB -> depth metres
ProbFn = Callable[[np.ndarray], np.ndarray]                # left RGB -> instrument probability


@dataclass
class Result:
    depth: np.ndarray
    prob: np.ndarray
    mask: np.ndarray
    tools: list[Tool]
    distance: float                        # metres, NaN when no instrument or no map yet
    map_points: int
    map_refreshed: bool
    ms: dict[str, float] = field(default_factory=dict)


class ProximityPipeline:
    def __init__(self, K: np.ndarray, depth_fn: DepthFn, prob_fn: ProbFn, map_every: int = 10,
                 live_tissue: bool = True, live_stride: int = 2, live_dilate_px: int = 12, erode_px: int = 1,
                 voxel_m: float = 0.5e-3, min_weight: float = 3.0, sync_cuda: bool = True):
        """`live_dilate_px`: margin around the instrument mask excluded from the current frame's tissue (the map
        always uses 5 px, as in Stage 5); `erode_px`: instrument-mask erosion before measuring."""
        self.K, self.depth_fn, self.prob_fn = K, depth_fn, prob_fn
        self.map_every, self.min_weight = map_every, min_weight
        self.live_tissue, self.live_stride, self.live_dilate_px = live_tissue, live_stride, live_dilate_px
        self.erode_px = erode_px
        self.tsdf = TSDFMap(K, voxel_m)
        self.map: TissueModel | None = None
        self.frames = 0
        self.sync = _cuda_sync if sync_cuda else (lambda: None)

    def _lap(self, ms: dict, key: str, t0: float) -> float:
        self.sync()
        t = time.perf_counter()
        ms[key] = 1000 * (t - t0)
        return t

    def step(self, left: np.ndarray, right: np.ndarray, T_wc: np.ndarray) -> Result:
        ms: dict[str, float] = {}
        t = t_start = time.perf_counter()
        depth = self.depth_fn(left, right)
        t = self._lap(ms, "depth", t)
        prob = self.prob_fn(left)
        mask = prob > 0.5
        t = self._lap(ms, "segmentation", t)
        tissue = tissue_depth(depth, mask)
        self.tsdf.integrate(left, tissue, T_wc)
        self.frames += 1
        refreshed = self.frames % self.map_every == 0 or self.map is None
        if refreshed:
            pts = self.tsdf.points(self.min_weight)
            self.map = TissueModel(pts) if len(pts) else None
        t = self._lap(ms, "fusion", t)
        model = self.map
        if self.live_tissue:
            model = TissueModel.merge(self.map, live_tissue(depth, mask, self.K, T_wc, self.live_dilate_px,
                                                            self.live_stride))
        tools = measure(depth, mask, self.K, T_wc, model, erode_px=self.erode_px)
        t = self._lap(ms, "proximity", t)
        ms["total"] = 1000 * (t - t_start)
        return Result(depth, prob, mask, tools, min_distance(tools),
                      0 if self.map is None else len(self.map), refreshed, ms)


def live_tissue(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, T_wc: np.ndarray, dilate_px: int,
                stride: int) -> TissueModel:
    """The current frame's own tissue pixels (outside the dilated instrument mask), every `stride`-th pixel."""
    Ks = K.copy()
    Ks[:2] /= stride     # pixel (u, v) of the full image is (u / s, v / s) of the subsampled one
    return TissueModel(backproject(tissue_depth(depth, mask, dilate_px)[::stride, ::stride], Ks, T_wc)[0])


def _cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
