"""Feed-forward camera poses with VGGT (Wang et al., CVPR 2025).

Only the aggregator and camera head run (no depth / point heads), in bfloat16, so ~50 frames fit in 8 GB.
Images are resized to VGGT's 518-px width with the height rounded to a multiple of 14, and the predicted
intrinsics are mapped back to the original resolution so they can be compared with the calibration.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGTPose:
    name = "vggt"

    def __init__(self, device: str = "cuda", dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device=device, dtype=dtype).eval()

    @staticmethod
    def _size(h: int, w: int) -> tuple[int, int]:
        nw = 518
        return round(h * nw / w / 14) * 14, nw

    @torch.inference_mode()
    def __call__(self, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 images (same size) -> (camera-to-world (N,4,4), intrinsics (N,3,3) at input resolution)."""
        h, w = images[0].shape[:2]
        nh, nw = self._size(h, w)
        batch = np.stack([cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA) for im in images])
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device, self.dtype)[None] / 255.0
        tokens, ps_idx = self.model.aggregator(x)
        pose_enc = self.model.camera_head(tokens)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc.float(), (nh, nw))
        T_cw = np.tile(np.eye(4), (len(images), 1, 1))
        T_cw[:, :3, :4] = extr[0].cpu().numpy()
        K = intr[0].cpu().numpy().astype(np.float64)
        K[:, 0] *= w / nw
        K[:, 1] *= h / nh
        return np.linalg.inv(T_cw), K
