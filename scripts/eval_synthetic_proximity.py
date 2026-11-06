"""Absolute instrument-tissue distance accuracy on synthetic stereo scenes with known geometry.

    python scripts/eval_synthetic_proximity.py data/scenes/stereomis_P2_1_tools

Tissue: real instrument-free P2_1 frames (their mesh is the ground truth); instrument: an 8 mm capsule placed
at exact gaps (`sr3d.eval.synthetic`). Each trial runs the real `ProximityPipeline` with the node's settings:
10 instrument-free frames (the map sees the tissue), then 10 frames with the instrument at the gap; the
distance on the last frame is scored. Masks: rendered ground truth, and SegFormer on the rendered image.
Nothing is tuned on these scenes.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sr3d.data.scene import Scene
from sr3d.depth.raft import RAFTStereo
from sr3d.eval import synthetic as syn
from sr3d.ros.pipeline import ProximityPipeline
from sr3d.segmentation.model import InstrumentSegmenter

DEPTH_SCALE = 100_000


def placements(scene: syn.Scene, n: int, rng, elevation_deg=(20, 40)):
    """Tissue points spread over the central image area; axis rises towards the nearest image border."""
    h, w = scene.h, scene.w
    left, _, _, pts, gid = scene.render(None, noise=0, blur=0)
    out = []
    grid = [(u, v) for v in np.linspace(0.3, 0.7, 3) for u in np.linspace(0.3, 0.7, 3)]
    for fu, fv in [grid[i] for i in rng.permutation(len(grid))[:n]]:
        u, v = int(fu * w), int(fv * h)
        p = pts[v, u]
        # surface normal from neighbouring hit points, oriented towards the camera
        n_ = np.cross(pts[v, u + 3] - pts[v, u - 3], pts[v + 3, u] - pts[v - 3, u])
        n_ /= np.linalg.norm(n_)
        if n_ @ -p < 0:
            n_ = -n_
        # lateral direction towards the nearest border, in the image plane (camera x / y)
        cands = [np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 1.0, 0])]
        dist = [u, w - u, v, h - v]
        lat = cands[int(np.argmin(dist))]
        el = np.deg2rad(rng.uniform(*elevation_deg))
        axis = np.cos(el) * lat + np.sin(el) * np.array([0, 0, -1.0])     # -z: towards the camera
        out.append((p, n_, axis / np.linalg.norm(axis), (u, v)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--cache", default="outputs/fusion")
    ap.add_argument("--seg", default="outputs/segmentation/segformer_b2_fluor/best")
    ap.add_argument("--frames", type=int, nargs="+", default=[30, 290], help="instrument-free tissue frames")
    ap.add_argument("--gaps-mm", type=float, nargs="+", default=[0.5, 1, 2, 3, 5, 7.5, 10])
    ap.add_argument("--placements", type=int, default=4)
    ap.add_argument("--out", default="outputs/synthetic")
    a = ap.parse_args()

    src = Scene(a.scene)
    K, B = src.K, src.meta.baseline_m
    cache = Path(a.cache) / src.meta.name
    raft = RAFTStereo()
    seg = InstrumentSegmenter(a.seg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for fi in a.frames:
        fr = src[fi]
        assert not fr.tool_mask.any(), f"frame {fi} has an instrument"
        depth0 = cv2.imread(str(cache / "depth_raft" / f"{fi:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
        sc = syn.Scene(*syn.tissue_mesh(depth0, fr.rgb, K), K, B, depth0.shape, seed=fi)
        L0, R0, *_ = sc.render(None)
        d_free = raft.depth(L0, R0, K[0, 0], B)
        for pi, (p, n_, axis, uv) in enumerate(placements(sc, a.placements, rng)):
            for gap_mm in a.gaps_mm:
                cap = sc.place(p, n_, axis, gap_mm * 1e-3)
                true_gap = float(sc.tissue_distance(cap.surface_samples()).min())
                L, R, gt_mask, pts, gid = sc.render(cap)
                if gt_mask.sum() < 500:
                    print(f"frame {fi} placement {pi}: instrument barely visible, skipped")
                    break
                vis = syn.visible_gap(sc, pts, gid)
                d_tool = raft.depth(L, R, K[0, 0], B)
                true_z = pts[..., 2]
                tz = gt_mask & (d_tool > 0)
                tool_depth_err = float(np.median(np.abs(d_tool[tz] - true_z[tz]))) if tz.any() else np.nan
                prob = seg.prob(L)
                row = {"frame": fi, "placement": pi, "gap_mm": gap_mm, "true_gap_mm": true_gap * 1e3,
                       "visible_gap_mm": vis * 1e3, "tool_px": int(gt_mask.sum()),
                       "tool_depth_abs_err_mm": tool_depth_err * 1e3,
                       "seg_iou": float((gt_mask & (prob > 0.5)).sum() / max((gt_mask | (prob > 0.5)).sum(), 1))}
                masks = {"gt_mask": gt_mask.astype(np.float32), "segformer": prob}
                # variants: visible surface only, and the capsule prior with the true radius (4 mm) and wrong
                # ones (3.5 / 4.5 mm) to show sensitivity to a mis-specified instrument
                # "capsule" = curvature fit only, "offset" = offset model only, "tiered" = fit else offset (deployed)
                variants = [("gt_mask", "gt_mask", None, None), ("segformer", "segformer", None, None),
                            ("gt_mask_capsule", "gt_mask", 4e-3, "fit"), ("segformer_capsule", "segformer", 4e-3, "fit"),
                            ("gt_mask_offset", "gt_mask", 4e-3, "offset"), ("segformer_offset", "segformer", 4e-3, "offset"),
                            ("gt_mask_tiered", "gt_mask", 4e-3, "tiered"), ("segformer_tiered", "segformer", 4e-3, "tiered"),
                            ("gt_mask_tiered_r3.5", "gt_mask", 3.5e-3, "tiered"), ("gt_mask_tiered_r4.5", "gt_mask", 4.5e-3, "tiered")]
                for name, mk, radius, mode in variants:
                    seq = [(L0, d_free)] * 10 + [(L, d_tool)] * 10
                    depths = iter(d for _, d in seq)
                    pipe = ProximityPipeline(K, lambda left, right: next(depths),
                                             lambda left, m=masks[mk]: m if left is L else
                                             np.zeros(left.shape[:2], np.float32), sync_cuda=False, tool_radius=radius,
                                             tool_mode=mode or "tiered")
                    for img, _ in seq:
                        res = pipe.step(img, None, np.eye(4))
                    row[f"{name}_mm"] = res.distance * 1e3
                    if radius:
                        row[f"{name}_fitted"] = int(sum(t.capsule is not None for t in res.tools))
                rows.append(row)
                print(f"frame {fi} pl {pi} gap {gap_mm:4.1f}: true {row['true_gap_mm']:.3f}  visible "
                      f"{row['visible_gap_mm']:.2f}  measured gt-mask {row['gt_mask_mm']:.2f}  segformer "
                      f"{row['segformer_mm']:.2f}  capsule gt-mask {row['gt_mask_capsule_mm']:.2f}  segformer "
                      f"{row['segformer_capsule_mm']:.2f}  tiered {row['gt_mask_tiered_mm']:.2f} mm  (tool depth err {row['tool_depth_abs_err_mm']:.2f} mm, "
                      f"seg IoU {row['seg_iou']:.2f})", flush=True)
                if pi == 0 and gap_mm == a.gaps_mm[0]:
                    cv2.imwrite(str(out / f"example_f{fi}.png"), cv2.cvtColor(np.concatenate([L, R], 1), cv2.COLOR_RGB2BGR))

    def summary(key, ref):
        e = np.array([r[key] - r[ref] for r in rows if np.isfinite(r[key])])
        miss = sum(not np.isfinite(r[key]) for r in rows)
        return {"mae_mm": float(np.mean(np.abs(e))), "bias_mm": float(np.mean(e)),
                "median_abs_mm": float(np.median(np.abs(e))), "p90_abs_mm": float(np.quantile(np.abs(e), 0.9)),
                "n": int(len(e)), "missing": int(miss)}
    res = {"trials": len(rows)}
    for key in [k for k in rows[0] if k.endswith("_mm") and k.startswith(("gt_mask", "segformer"))]:
        res[key] = {"vs_true_gap": summary(key, "true_gap_mm"), "vs_visible_gap": summary(key, "visible_gap_mm")}
        res[key]["per_gap"] = {str(g): summary_sel(rows, key, g) for g in a.gaps_mm}
    res["visible_minus_true_mm"] = float(np.mean([r["visible_gap_mm"] - r["true_gap_mm"] for r in rows]))
    res["tool_depth_abs_err_mm_median"] = float(np.nanmedian([r["tool_depth_abs_err_mm"] for r in rows]))
    res["seg_iou_mean"] = float(np.mean([r["seg_iou"] for r in rows]))
    (out / "synthetic_proximity.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1))
    print(json.dumps(res, indent=1))


def summary_sel(rows, key, gap):
    sel = [r for r in rows if r["gap_mm"] == gap and np.isfinite(r[key])]
    if not sel:
        return {}
    e = np.array([r[key] - r["true_gap_mm"] for r in sel])
    return {"n": len(sel), "mean_measured_mm": float(np.mean([r[key] for r in sel])),
            "bias_mm": float(e.mean()), "mae_mm": float(np.abs(e).mean()),
            "visible_gap_mm": float(np.mean([r["visible_gap_mm"] for r in sel]))}


if __name__ == "__main__":
    main()
