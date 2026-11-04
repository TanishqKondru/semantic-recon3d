"""Monocular relative depth with Depth Anything V2 (Yang et al., NeurIPS 2024).

The network predicts affine-invariant inverse depth, so metric depth needs a scale and shift;
`align_inverse_depth` fits them against a sparse or dense metric reference (e.g. stereo).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingV2:
    def __init__(self, size: str = "base", device: str = "cuda"):
        self.name = f"dav2_{size}"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(MODELS[size])
        self.model = AutoModelForDepthEstimation.from_pretrained(MODELS[size], dtype=torch.float16).eval().to(device)

    @torch.inference_mode()
    def relative_inverse_depth(self, rgb: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device, torch.float16)
        pred = self.model(**inputs).predicted_depth
        pred = F.interpolate(pred[:, None].float(), size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return pred[0, 0].cpu().numpy()


def align_inverse_depth(rel_inv: np.ndarray, ref_depth: np.ndarray, mask: np.ndarray | None = None,
                        robust_iters: int = 3) -> tuple[np.ndarray, float, float]:
    """Fit inv_depth = s * rel_inv + t to the reference with iteratively trimmed least squares.

    Returns (metric depth, s, t). Pixels where the aligned inverse depth is non-positive get depth 0.
    """
    valid = ref_depth > 0
    if mask is not None:
        valid &= mask
    x, y = rel_inv[valid].astype(np.float64), 1.0 / ref_depth[valid].astype(np.float64)
    keep = np.ones_like(x, dtype=bool)
    for _ in range(robust_iters):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        (s, t), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        r = np.abs(s * x + t - y)
        keep = r < np.percentile(r, 90)
    inv = s * rel_inv + t
    depth = np.where(inv > 1e-6, 1.0 / np.maximum(inv, 1e-6), 0.0).astype(np.float32)
    return depth, float(s), float(t)
