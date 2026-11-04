"""Learned stereo via RAFT optical flow (Teed & Deng, ECCV 2020, torchvision weights).

On a rectified pair the horizontal flow left->right is -disparity, and the vertical component should be ~0,
which doubles as a rectification-quality check. Invalid pixels are rejected with a left-right
(forward-backward) consistency test.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from ..geometry import disparity_to_depth


class RAFTStereo:
    name = "raft"

    def __init__(self, device: str = "cuda", iters: int = 12, lr_check_px: float = 1.0):
        self.device = device
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval().to(device)
        self.iters = iters
        self.lr_check_px = lr_check_px
        self.last_vertical_flow = None

    def _prep(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).float()[None].to(self.device)
        return t / 127.5 - 1.0

    @torch.inference_mode()
    def _flow(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.model(a, b, num_flow_updates=self.iters)[-1]

    @torch.inference_mode()
    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        h, w = left.shape[:2]
        if h % 8 or w % 8:
            raise ValueError("RAFT needs image sides divisible by 8")
        L, R = self._prep(left), self._prep(right)
        # one batched call for both directions
        flows = self._flow(torch.cat([L, R]), torch.cat([R, L]))
        fwd, bwd = flows[0], flows[1]
        disp = -fwd[0]
        self.last_vertical_flow = fwd[1].abs().cpu().numpy()

        if self.lr_check_px > 0:
            # sample the backward flow where each left pixel lands in the right image
            ys, xs = torch.meshgrid(torch.arange(h, device=L.device), torch.arange(w, device=L.device), indexing="ij")
            gx = (xs + fwd[0]) / (w - 1) * 2 - 1
            gy = (ys + fwd[1]) / (h - 1) * 2 - 1
            back = F.grid_sample(bwd[None], torch.stack([gx, gy], -1)[None], align_corners=True)[0]
            cycle = torch.linalg.norm(fwd + back, dim=0)
            disp[cycle > self.lr_check_px] = 0
        disp[disp <= 0] = 0
        return disp.float().cpu().numpy()

    def depth(self, left, right, fx: float, baseline_m: float) -> np.ndarray:
        return disparity_to_depth(self.disparity(left, right), fx, baseline_m)
