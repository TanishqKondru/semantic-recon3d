"""Fuse a scene into a TSDF tissue map and score it on held-out frames.

Inputs come from scripts/prepare_fusion.py. Every `--holdout`-th frame is left out of fusion and used
for scoring (rendered vs observed depth and colour, coverage). Instrument pixels are removed before
integration using one of several masks; `ghost_pct` is the share of map vertices that ground-truth masks
say were built from instrument pixels (multi-view vote > 0.5), i.e. moving-tool geometry left in the map.

    python scripts/run_fusion.py data/scenes/stereomis_P2_1_tools
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from sr3d.data.scene import Scene
from sr3d.eval.fusion_metrics import MeshRenderer, heldout_scores
from sr3d.fusion.labels import label_iou, vote
from sr3d.data.poses import read_tum, write_tum
from sr3d.eval.pose_metrics import evaluate
from sr3d.fusion.tsdf import TSDFMap, tissue_depth
from sr3d.pose.stereo_vo import StereoVO

DEPTH_SCALE = 100_000


def vo_poses(scene, depth, tool_masks, cache):
    """Stereo VO over every frame (cached depth, predicted masks), anchored to the first kinematic pose."""
    path = cache / "poses_stereo_vo.txt"
    if path.exists():
        return read_tum(path)[1]
    frames = [scene[i] for i in range(len(scene))]
    for f, m in zip(frames, tool_masks):
        f.tool_mask = m
    queue = iter(depth)
    T, inliers = StereoVO(lambda left, right: next(queue))(frames, scene.K)
    T = scene.poses[0] @ T
    write_tum(path, np.arange(len(T), dtype=float), T)
    m = evaluate(T, scene.poses)
    print(f"stereo VO over {len(T)} frames: ATE {m['ate_rmse_mm']:.2f} mm (Sim3, scale {m['scale']:.3f}), "
          f"failed steps {int((inliers[1:] == 0).sum())}", flush=True)
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--cache", default="outputs/fusion")
    ap.add_argument("--masks", nargs="+", default=["none", "gt", "segformer", "sam2"])
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--voxel-mm", type=float, default=0.5)
    ap.add_argument("--trunc-voxels", type=float, default=5.0)
    ap.add_argument("--poses", default="kinematics", choices=["kinematics", "stereo_vo"],
                    help="camera poses for fusion and held-out rendering (stereo_vo = vision only)")
    a = ap.parse_args()

    scene = Scene(a.scene)
    cache = Path(a.cache) / scene.meta.name
    K, w, h = scene.K, scene.meta.width, scene.meta.height
    n = len(scene)
    rgb = [scene[i].rgb for i in range(n)]
    gt = [scene[i].tool_mask for i in range(n)]
    depth = [cv2.imread(str(cache / "depth_raft" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
             for i in range(n)]
    prob = {k: [cv2.imread(str(cache / f"prob_{k}" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) / 255.0 for i in range(n)]
            for k in ("segformer", "sam2")}
    masks = {"none": [None] * n, "gt": gt, **{k: [p > 0.5 for p in v] for k, v in prob.items()}}
    poses = scene.poses
    if a.poses == "stereo_vo":
        poses = vo_poses(scene, depth, masks["sam2"], cache)
    test = [i for i in range(n) if i % a.holdout == a.holdout // 2]
    train = [i for i in range(n) if i not in test]
    print(f"{scene.meta.name}: {len(train)} fusion frames, {len(test)} held-out", flush=True)

    suffix = "" if a.poses == "kinematics" else f"_{a.poses}"
    results = {}
    for name in a.masks:
        t0 = time.time()
        tsdf = TSDFMap(K, a.voxel_mm / 1000, a.trunc_voxels)
        for i in train:
            tsdf.integrate(rgb[i], tissue_depth(depth[i], masks[name][i]), poses[i])
        mesh = tsdf.mesh()
        build_s = time.time() - t0
        o3d.io.write_triangle_mesh(str(cache / f"tsdf_{name}{suffix}.ply"), mesh)

        renderer = MeshRenderer(mesh)
        scores = []
        for i in test:
            rd, rc = renderer.render(K, poses[i], w, h)
            scores.append(heldout_scores(rd, rc, depth[i], rgb[i], ~gt[i]))
        pts = np.asarray(mesh.vertices)
        ref = vote(pts, [(gt[i].astype(float), depth[i], poses[i]) for i in train], K)
        row = {k: float(np.nanmean([s[k] for s in scores])) for k in scores[0]}
        row.update(vertices=len(pts), ghost_pct=100 * float(np.mean(ref.prob[ref.seen > 0] > 0.5)),
                   unseen_pct=100 * float(np.mean(ref.seen == 0)), build_s=build_s)
        # 3D labels from predicted masks, scored against the ground-truth vote on the same points
        for k, probs in prob.items():
            v = vote(pts, [(probs[i], depth[i], poses[i]) for i in train], K)
            row[f"label_iou_{k}_vote"] = label_iou(v.prob, ref.prob)
            row[f"label_iou_{k}_last"] = label_iou(v.last, ref.prob)
        results[name] = row
        print(f"{name:10s} depth {row['depth_mm']:.2f} mm  PSNR {row['psnr']:.2f} dB  coverage {row['coverage']:.3f}  "
              f"ghost {row['ghost_pct']:.2f}%  {row['vertices']} verts  {build_s:.0f} s", flush=True)
        print("           3D label IoU " + "  ".join(f"{k[10:]} {v:.3f}" for k, v in row.items() if k.startswith("label_iou")),
              flush=True)

    (cache / f"tsdf_summary{suffix}.json").write_text(json.dumps(
        {"args": vars(a), "train": len(train), "test": test, "results": results}, indent=1))


if __name__ == "__main__":
    main()
