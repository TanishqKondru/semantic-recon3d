"""Classical stereo baseline: OpenCV semi-global block matching with left-right consistency."""
from __future__ import annotations

import cv2
import numpy as np

from ..geometry import disparity_to_depth


class SGBMStereo:
    name = "sgbm"

    def __init__(self, num_disparities: int = 128, block_size: int = 5, lr_check_px: float = 1.0):
        num_disparities = int(np.ceil(num_disparities / 16) * 16)
        common = dict(minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
                      P1=8 * 3 * block_size**2, P2=32 * 3 * block_size**2, disp12MaxDiff=-1,
                      uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                      mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left_matcher = cv2.StereoSGBM_create(**common)
        self.lr_check_px = lr_check_px

    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """RGB uint8 rectified pair -> float32 disparity in pixels, 0 where invalid."""
        gl, gr = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (left, right))
        d_l = self.left_matcher.compute(gl, gr).astype(np.float32) / 16.0
        if self.lr_check_px > 0:
            # right-view disparity via the flipped-image trick: match mirrored right against mirrored left
            d_r = self.left_matcher.compute(gr[:, ::-1].copy(), gl[:, ::-1].copy()).astype(np.float32)[:, ::-1] / 16.0
            h, w = d_l.shape
            u = np.arange(w)[None].repeat(h, 0)
            ur = np.clip(np.round(u - d_l).astype(int), 0, w - 1)
            back = np.take_along_axis(d_r, ur, axis=1)
            d_l[np.abs(d_l - back) > self.lr_check_px] = 0
        d_l[d_l <= 0] = 0
        return d_l

    def depth(self, left, right, fx: float, baseline_m: float) -> np.ndarray:
        return disparity_to_depth(self.disparity(left, right), fx, baseline_m)
