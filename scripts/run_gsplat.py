"""Fit semantic Gaussian splatting to a scene and score it on the same held-out frames as the TSDF map.

Needs scripts/prepare_fusion.py and scripts/run_fusion.py first (the masked TSDF mesh initialises the
Gaussians). Instrument pixels (SAM 2 mask, dilated) are excluded from training; scoring uses
ground-truth tissue pixels. Each Gaussian gets an instrument probability by multi-view voting.

    python scripts/run_gsplat.py data/scenes/stereomis_P2_1_tools
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from sr3d.data.scene import Scene
from sr3d.eval.fusion_metrics import heldout_scores
from sr3d.fusion.gaussians import GaussianMap, camera_radius
from sr3d.fusion.labels import vote
from sr3d.fusion.tsdf import tissue_depth

DEPTH_SCALE = 100_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--cache", default="outputs/fusion")
    ap.add_argument("--init", default="tsdf_sam2.ply")
    ap.add_argument("--mask", default="sam2", choices=["sam2", "segformer"])
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--depth-weight", type=float, default=0.05)
    ap.add_argument("--max-init", type=int, default=150_000)
    ap.add_argument("--tag", default="gsplat")
    a = ap.parse_args()

    torch.manual_seed(0)
    scene = Scene(a.scene)
    cache = Path(a.cache) / scene.meta.name
    K, w, h, n = scene.K, scene.meta.width, scene.meta.height, len(scene)
    rgb = [scene[i].rgb for i in range(n)]
    gt = [scene[i].tool_mask for i in range(n)]
    depth = [cv2.imread(str(cache / "depth_raft" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
             for i in range(n)]
    prob = [cv2.imread(str(cache / f"prob_{a.mask}" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) / 255.0 for i in range(n)]
    test = [i for i in range(n) if i % a.holdout == a.holdout // 2]
    train = [i for i in range(n) if i not in test]

    mesh = o3d.io.read_triangle_mesh(str(cache / a.init))
    pts, cols = np.asarray(mesh.vertices), np.clip(np.asarray(mesh.vertex_colors), 0, 1)
    keep = np.random.default_rng(0).permutation(len(pts))[:a.max_init]
    gs = GaussianMap(pts[keep], cols[keep], camera_radius(scene.poses[train]))
    # tissue = not instrument, with the same dilated border the TSDF fusion uses
    frames = [(rgb[i], depth[i], tissue_depth(np.ones((h, w), np.float32), prob[i] > 0.5) > 0, scene.poses[i])
              for i in train]
    print(f"{scene.meta.name}: {len(keep)} initial gaussians, {len(train)} frames, {a.iters} steps", flush=True)
    t0 = time.time()
    log = gs.fit(frames, K, w, h, iters=a.iters, depth_weight=a.depth_weight)
    fit_s = time.time() - t0

    t0 = time.time()
    scores = []
    for i in test:
        rd, rc = gs.render_numpy(K, scene.poses[i], w, h)
        scores.append(heldout_scores(rd, rc, depth[i], rgb[i], ~gt[i]))
    render_ms = 1000 * (time.time() - t0) / len(test)
    centers = gs.centers()
    v = vote(centers, [(prob[i], depth[i], scene.poses[i]) for i in train], K)
    row = {k: float(np.nanmean([s[k] for s in scores])) for k in scores[0]}
    row.update(gaussians=len(centers), fit_s=fit_s, render_ms=render_ms,
               instrument_pct=100 * float(np.mean(v.prob[v.seen > 0] > 0.5)))
    print(f"{a.tag}: depth {row['depth_mm']:.2f} mm  PSNR {row['psnr']:.2f} dB  coverage {row['coverage']:.3f}  "
          f"{row['gaussians']} gaussians  fit {fit_s / 60:.1f} min  render {render_ms:.0f} ms", flush=True)
    torch.save({k: p.detach().cpu() for k, p in gs.params.items()}, cache / f"{a.tag}.pt")
    (cache / f"{a.tag}_summary.json").write_text(json.dumps({"args": vars(a), "results": row, "log": log}, indent=1))


if __name__ == "__main__":
    main()
