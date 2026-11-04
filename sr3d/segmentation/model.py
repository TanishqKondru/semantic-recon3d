"""Binary instrument segmentation with SegFormer (Xie et al., NeurIPS 2021).

The MiT encoder is ImageNet-pretrained; the all-MLP decoder is trained from scratch. Logits come
out at 1/4 resolution and are upsampled bilinearly to the input size.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .data import IMAGENET_MEAN, IMAGENET_STD

LABELS = {0: "tissue", 1: "instrument"}


def build_segformer(encoder: str = "nvidia/mit-b2") -> SegformerForSemanticSegmentation:
    return SegformerForSemanticSegmentation.from_pretrained(
        encoder, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})


def logits_full(model, x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) normalised images -> (B, 2, H, W) logits."""
    return F.interpolate(model(pixel_values=x).logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def ce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy plus soft Dice on the instrument class (instruments are ~12% of tool-frame pixels)."""
    ce = F.cross_entropy(logits, target)
    p = logits.float().softmax(1)[:, 1]
    t = target.float()
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return ce + dice


class InstrumentSegmenter:
    """Inference wrapper for a trained checkpoint: RGB uint8 frame -> bool instrument mask."""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.device = device
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint).eval().to(device)

    @torch.inference_mode()
    def prob(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 frame -> (H, W) float32 instrument probability."""
        x = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            return logits_full(self.model, x).float().softmax(1)[0, 1].cpu().numpy()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.prob(rgb) > 0.5
