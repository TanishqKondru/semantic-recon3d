"""Cache the per-frame inputs of semantic fusion for a scene: stereo depth and instrument probabilities.

Writes, for every frame of the scene (the reconstruction scene P2_1 was held out of segmentation training):
  depth_raft/000123.png      RAFT stereo depth, uint16, metres = value / 100000
  prob_segformer/000123.png  SegFormer instrument probability, uint8 (p * 255)
  prob_sam2/000123.png       SAM 2 propagation prompted by SegFormer every `--every` frames, uint8

    python scripts/prepare_fusion.py data/scenes/stereomis_P2_1_tools
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sr3d.data.scene import Scene
from sr3d.depth.raft import RAFTStereo
from sr3d.segmentation.model import InstrumentSegmenter
from sr3d.segmentation.tracking import keyframes, load_sam2, track

DEPTH_SCALE = 100_000  # same encoding as scripts/run_depth.py


def write_prob(path: Path, prob: np.ndarray):
    cv2.imwrite(str(path), np.round(np.clip(prob, 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--out", default="outputs/fusion")
    ap.add_argument("--seg", default="outputs/segmentation/segformer_b2_fluor/best")
    ap.add_argument("--sam", default="large")
    ap.add_argument("--every", type=int, default=50, help="SAM 2 re-prompt interval in scene frames")
    ap.add_argument("--skip-sam2", action="store_true", help="depth and SegFormer only (enough for proximity)")
    a = ap.parse_args()

    scene = Scene(a.scene)
    out = Path(a.out) / scene.meta.name
    for d in ("depth_raft", "prob_segformer") + (() if a.skip_sam2 else ("prob_sam2",)):
        (out / d).mkdir(parents=True, exist_ok=True)
    frames = [scene[i] for i in range(len(scene))]
    t0 = time.time()

    raft = RAFTStereo()
    for i, f in enumerate(frames):
        d = raft.depth(f.rgb, f.right, scene.K[0, 0], scene.meta.baseline_m)
        cv2.imwrite(str(out / "depth_raft" / f"{i:06d}.png"), np.clip(d * DEPTH_SCALE, 0, 65535).astype(np.uint16))
    del raft
    torch.cuda.empty_cache()
    print(f"depth: {len(frames)} frames, {(time.time() - t0) / 60:.1f} min", flush=True)

    seg = InstrumentSegmenter(a.seg)
    seg_prob = [seg.prob(f.rgb) for f in frames]
    for i, p in enumerate(seg_prob):
        write_prob(out / "prob_segformer" / f"{i:06d}.png", p)
    del seg
    torch.cuda.empty_cache()
    print(f"segformer done, {(time.time() - t0) / 60:.1f} min", flush=True)

    if a.skip_sam2:
        return
    sam = load_sam2(a.sam)
    prompts = {i: seg_prob[i] > 0.5 for i in keyframes(len(frames), a.every)}
    for i, p in enumerate(track(sam, [f.rgb for f in frames], prompts)):
        write_prob(out / "prob_sam2" / f"{i:06d}.png", p)
    print(f"sam2 done, {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
