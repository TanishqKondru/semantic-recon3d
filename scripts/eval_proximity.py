"""Score the instrument-to-tissue distance of the online pipeline against a ground-truth-mask reference.

There is no measured tool-tissue distance in StereoMIS, so the reference isolates what the pipeline adds:
same RAFT depth, but ground-truth instrument masks and a tissue map fused from all frames with those masks.
Variants replay cached depth and masks (scripts/prepare_fusion.py) through `ProximityPipeline`:

  online_segformer  the deployed node: SegFormer masks, causal map (frames up to now) + current frame's tissue
  online_map_only   as above without the current frame's tissue
  online_every1     as online_segformer with the map's k-d tree rebuilt every frame instead of every 10th
  online_gt         as online_segformer with ground-truth masks: the cost of segmentation errors
  offline_segformer SegFormer masks, map from all frames: the cost of causality (tissue not yet seen)
  live_frame        no map, tissue = current frame's own tissue pixels only

`--sweep` instead tunes the instrument margins of online_segformer (live-tissue dilation, instrument erosion);
run it on the tuning scene (P1), then evaluate the test scene (P2_1) once with the chosen values.

    python scripts/eval_proximity.py data/scenes/stereomis_P1_tools --sweep
    python scripts/eval_proximity.py data/scenes/stereomis_P2_1_tools      (defaults = margins chosen on P1)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sr3d.data.scene import Scene
from sr3d.fusion.tsdf import TSDFMap, tissue_depth
from sr3d.ros.pipeline import ProximityPipeline, live_tissue
from sr3d.ros.proximity import TissueModel, measure, min_distance

DEPTH_SCALE = 100_000
ALARMS_MM = (2.0, 5.0)


def full_map(K, rgb, depth, masks, poses):
    tsdf = TSDFMap(K)
    for c, d, m, T in zip(rgb, depth, masks, poses):
        tsdf.integrate(c, tissue_depth(d, m), T)
    return TissueModel(tsdf.points())


def replay(K, rgb, depth, masks, poses, **kw):
    """Feed cached depth / masks through the online pipeline in time order -> (results, map after each frame)."""
    i = iter(range(len(rgb)))
    k = {"i": 0}

    def depth_fn(left, right):
        k["i"] = next(i)
        return depth[k["i"]]

    pipe = ProximityPipeline(K, depth_fn, lambda left: masks[k["i"]].astype(np.float32), sync_cuda=False, **kw)
    results, maps = [], []
    for c, T in zip(rgb, poses):
        results.append(pipe.step(c, None, T))
        maps.append(pipe.map)
    return results, maps


def sweep(K, depth, masks, poses, maps, ref, has_tool, out, dilates, erodes, tag=""):
    """Re-measure on the causal map snapshots for each margin setting (the map itself does not depend on them)."""
    rows = []
    for dil in dilates:
        for ero in erodes:
            d = np.array([min_distance(measure(depth[i], masks[i], K, poses[i],
                                               TissueModel.merge(maps[i], live_tissue(depth[i], masks[i], K, poses[i],
                                                                                      dil, 2)), erode_px=ero))
                          for i in range(len(depth))])
            row = {"live_dilate_px": dil, "erode_px": ero, **score(d, ref, has_tool)}
            rows.append(row)
            print(f"dilate {dil} erode {ero}: MAE {row['mae_mm']:.3f} mm  hover {row['mae_hover_mm']:.3f} "
                  f"({row['frames_hover']} fr)  contact {row['mae_contact_mm']:.3f}  median {row['median_mm']:.2f}  "
                  f"P90 {row['p90_mm']:.2f}  bias {row['bias_mm']:+.2f}  "
                  f"alarm<2mm R/P {row['alarm2_recall']:.2f}/{row['alarm2_precision']:.2f}", flush=True)
    best = min(rows, key=lambda r: r["mae_mm"])
    print(f"best by MAE: dilate {best['live_dilate_px']} erode {best['erode_px']} ({best['mae_mm']:.3f} mm)")
    (out / f"proximity_sweep{tag}.json").write_text(json.dumps({"rows": rows, "best": best}, indent=1))


def tip_errors(pred_tools, ref_tools, max_px=50):
    """Match predicted to reference tips (nearest within `max_px`) -> list of (pixel error, 3D error m)."""
    out = []
    for r in ref_tools:
        if r.tip_px is None or r.tip is None:
            continue
        cands = [p for p in pred_tools if p.tip_px is not None and p.tip is not None]
        if not cands:
            continue
        p = min(cands, key=lambda p: np.hypot(p.tip_px[0] - r.tip_px[0], p.tip_px[1] - r.tip_px[1]))
        e = float(np.hypot(p.tip_px[0] - r.tip_px[0], p.tip_px[1] - r.tip_px[1]))
        if e <= max_px:
            out.append((e, float(np.linalg.norm(p.tip - r.tip))))
    return out


def score(d, ref, has_tool):
    both = np.isfinite(d) & np.isfinite(ref)
    err = np.abs(d[both] - ref[both]) * 1e3
    row = {"frames_scored": int(both.sum()), "frames_ref": int(np.isfinite(ref).sum()),
           "detected_pct": 100 * float(both.sum() / max(np.isfinite(ref).sum(), 1)),
           "mae_mm": float(err.mean()), "median_mm": float(np.median(err)), "p90_mm": float(np.quantile(err, 0.9)),
           "bias_mm": float(np.mean(d[both] - ref[both]) * 1e3),
           "false_tool_frames": int((np.isfinite(d) & ~has_tool).sum())}
    # hover (reference >= 2 mm) and contact frames penalise opposite errors, so they are reported apart
    for part, sel in (("hover", ref[both] >= 2e-3), ("contact", ref[both] < 2e-3)):
        row[f"frames_{part}"] = int(sel.sum())
        row[f"mae_{part}_mm"] = float(err[sel].mean()) if sel.any() else float("nan")
    for thr in ALARMS_MM:
        p = np.nan_to_num(d, nan=np.inf) < thr / 1e3
        g = np.nan_to_num(ref, nan=np.inf) < thr / 1e3
        row[f"alarm{thr:g}_recall"] = float((p & g).sum() / max(g.sum(), 1))
        row[f"alarm{thr:g}_precision"] = float((p & g).sum() / max(p.sum(), 1))
        row[f"alarm{thr:g}_ref_frames"] = int(g.sum())
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--cache", default="outputs/fusion")
    ap.add_argument("--out", default="outputs/ros")
    ap.add_argument("--map-every", type=int, default=10)
    ap.add_argument("--live-dilate", type=int, default=12, help="instrument margin excluded from live tissue (px)")
    ap.add_argument("--erode", type=int, default=1, help="instrument-mask erosion before measuring (px)")
    ap.add_argument("--sweep", action="store_true", help="tune --live-dilate / --erode instead of evaluating")
    ap.add_argument("--sweep-dilate", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8])
    ap.add_argument("--sweep-erode", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--sweep-tag", default="", help="suffix for the sweep output file")
    a = ap.parse_args()

    scene = Scene(a.scene)
    cache = Path(a.cache) / scene.meta.name
    out = Path(a.out) / scene.meta.name
    out.mkdir(parents=True, exist_ok=True)
    K, n, poses = scene.K, len(scene), scene.poses
    rgb = [scene[i].rgb for i in range(n)]
    gt = [scene[i].tool_mask for i in range(n)]
    depth = [cv2.imread(str(cache / "depth_raft" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
             for i in range(n)]
    seg = [cv2.imread(str(cache / "prob_segformer" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for i in range(n)]

    # reference: fixed margins (defaults), ground-truth masks, map from all frames
    ref_map = full_map(K, rgb, depth, gt, poses)
    ref_tools = [measure(depth[i], gt[i], K, poses[i], ref_map) for i in range(n)]
    ref = np.array([min_distance(t) for t in ref_tools])
    has_tool = np.array([len(t) > 0 for t in ref_tools])
    print(f"reference: {int(np.isfinite(ref).sum())} frames with a distance, {int(has_tool.sum())} with an instrument, "
          f"median {np.nanmedian(ref) * 1e3:.2f} mm, map {len(ref_map)} points", flush=True)

    margins = {"live_dilate_px": a.live_dilate, "erode_px": a.erode}
    online, maps = replay(K, rgb, depth, seg, poses, map_every=a.map_every, **margins)
    if a.sweep:
        # the sweep's fast path must reproduce the pipeline exactly
        fast = [min_distance(measure(depth[i], seg[i], K, poses[i],
                                     TissueModel.merge(maps[i], live_tissue(depth[i], seg[i], K, poses[i],
                                                                            a.live_dilate, 2)), erode_px=a.erode))
                for i in range(n)]
        assert np.allclose(fast, [r.distance for r in online], equal_nan=True)
        sweep(K, depth, seg, poses, maps, ref, has_tool, out, a.sweep_dilate, a.sweep_erode, a.sweep_tag)
        return

    seg_map = full_map(K, rgb, depth, seg, poses)
    tools = {
        "online_segformer": [r.tools for r in online],
        "online_map_only": [r.tools for r in replay(K, rgb, depth, seg, poses, map_every=a.map_every,
                                                    live_tissue=False, **margins)[0]],
        "online_every1": [r.tools for r in replay(K, rgb, depth, seg, poses, map_every=1, **margins)[0]],
        "online_gt": [r.tools for r in replay(K, rgb, depth, gt, poses, map_every=a.map_every, **margins)[0]],
        "offline_segformer": [measure(depth[i], seg[i], K, poses[i], seg_map, erode_px=a.erode) for i in range(n)],
        "live_frame": [measure(depth[i], seg[i], K, poses[i], live_tissue(depth[i], seg[i], K, poses[i],
                                                                          a.live_dilate, 1), erode_px=a.erode)
                       for i in range(n)],
    }
    per_frame = {"ref": ref.tolist()}
    results = {}
    for name, tl in tools.items():
        d = np.array([min_distance(t) for t in tl])
        per_frame[name] = d.tolist()
        row = score(d, ref, has_tool)
        tips = [e for p, r in zip(tl, ref_tools) for e in tip_errors(p, r)]
        row["tips_matched"] = len(tips)
        row["tips_ref"] = sum(r.tip_px is not None and r.tip is not None for rt in ref_tools for r in rt)
        row["tip_px_median"] = float(np.median([e[0] for e in tips])) if tips else np.nan
        row["tip_mm_median"] = float(np.median([e[1] for e in tips]) * 1e3) if tips else np.nan
        results[name] = row
        print(f"{name:18s} MAE {row['mae_mm']:.2f} mm  median {row['median_mm']:.2f}  P90 {row['p90_mm']:.2f}  "
              f"bias {row['bias_mm']:+.2f}  detected {row['detected_pct']:.0f}%  "
              f"alarm<2mm R/P {row['alarm2_recall']:.2f}/{row['alarm2_precision']:.2f}  "
              f"tips {row['tips_matched']}/{row['tips_ref']} {row['tip_px_median']:.1f} px {row['tip_mm_median']:.2f} mm",
              flush=True)
    (out / "proximity_eval.json").write_text(json.dumps(
        {"args": vars(a), "results": results, "per_frame_m": per_frame}, indent=1))


if __name__ == "__main__":
    main()
