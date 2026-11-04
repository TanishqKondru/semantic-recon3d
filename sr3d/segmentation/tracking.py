"""Temporal instrument segmentation: SAM 2 video propagation prompted by per-frame SegFormer masks.

SegFormer sees each frame alone, so its masks flicker and drop parts of a tool for single frames.
SAM 2 (Ravi et al., 2024) carries an object through a memory bank, which gives temporally consistent
masks but drifts and never notices new tools on its own. Re-prompting SAM 2 with the SegFormer mask
every `every` frames combines the two: SegFormer decides *what* is an instrument, SAM 2 keeps it
consistent in between.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

SAM2_MODELS = {  # name -> (config inside the sam2 package, checkpoint repo, file)
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "facebook/sam2.1-hiera-base-plus", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt"),
}


def load_sam2(size: str = "small", device: str = "cuda"):
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2_video_predictor

    cfg, repo, fname = SAM2_MODELS[size]
    return build_sam2_video_predictor(cfg, hf_hub_download(repo, fname), device=device)


def keyframes(n_frames: int, every: int) -> list[int]:
    """Prompt frames: 0, every, 2*every, ...; every <= 0 means the first frame only."""
    return [0] if every <= 0 else list(range(0, n_frames, every))


def uncertainty(prob: np.ndarray) -> float:
    """Unsure pixels (0.2 < p < 0.8) per predicted instrument pixel; on P1 val frames it correlates
    with per-frame IoU at Spearman -0.53 (top quartile: mean IoU 0.54 vs 0.79)."""
    return float(((prob > 0.2) & (prob < 0.8)).sum() / max((prob > 0.5).sum(), 1))


def confident_keyframes(uncert: list[float], window: int, max_uncert: float = 0.25) -> list[int]:
    """One prompt per window: its most confident frame, skipped if even that is above max_uncert.

    A wrong prompt is worse than none, because SAM 2 propagates it to every neighbouring frame;
    between accepted prompts SAM 2 simply keeps tracking. The first window always gets a prompt.
    """
    keys = []
    for s in range(0, len(uncert), window):
        best = s + int(np.argmin(uncert[s:s + window]))
        if not keys or uncert[best] <= max_uncert:
            keys.append(best)
    return keys


@torch.inference_mode()
def track(predictor, frames: list[np.ndarray], prompts: dict[int, np.ndarray], both_ways: bool = False) -> list[np.ndarray]:
    """Propagate instrument masks through a clip; returns per-frame instrument probability (H, W).

    frames: RGB uint8 images in time order. prompts: frame index -> bool mask (all instruments as one
    object; an empty prompt is skipped because SAM 2 cannot condition on nothing). Frames no pass
    reaches come back 0. both_ways also propagates backwards from a fresh state and averages the
    two passes, so every frame gets its nearest prompt in either direction.
    """
    prompts = {k: m for k, m in prompts.items() if m.any()}
    h, w = frames[0].shape[:2]
    if not prompts:
        return [np.zeros((h, w), np.float32) for _ in frames]
    first = min(prompts)
    # a backward pass is always needed to cover frames before the first prompt
    passes = [False, True] if both_ways or first > 0 else [False]
    total = [np.zeros((h, w), np.float32) for _ in frames]
    count = np.zeros(len(frames))
    with tempfile.TemporaryDirectory() as d:  # the predictor reads a folder of JPEGs named by index
        for i, f in enumerate(frames):
            cv2.imwrite(str(Path(d) / f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(d, offload_video_to_cpu=True)
            for reverse in passes:
                for k, m in sorted(prompts.items()):
                    predictor.add_new_mask(state, k, obj_id=1, mask=m)
                start = max(prompts) if reverse else min(prompts)
                for idx, _, logits in predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                    if reverse and not both_ways and idx >= first:
                        continue
                    total[idx] += torch.sigmoid(logits[0, 0].float()).cpu().numpy()
                    count[idx] += 1
                predictor.reset_state(state)
    return [t / max(c, 1) for t, c in zip(total, count)]


def flicker(masks: list[np.ndarray]) -> float:
    """Mean (1 - IoU) between consecutive masks; frames where both are empty count as stable."""
    vals = []
    for a, b in zip(masks[:-1], masks[1:]):
        union = (a | b).sum()
        vals.append(0.0 if union == 0 else 1 - (a & b).sum() / union)
    return float(np.mean(vals)) if vals else 0.0
