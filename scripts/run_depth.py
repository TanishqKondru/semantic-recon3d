"""Run depth methods on a scene, save predictions and score them without ground-truth depth.

Scores:
  coverage        fraction of tissue pixels with a depth estimate
  warp_err        photometric error when frame i+gap is warped into frame i using the method's depth
                  and the GT kinematic poses (lower = geometrically more consistent), on that method's pixels
  warp_err_common same, restricted to pixels every method covers (fair comparison)
  geo_mm          median |z of frame-i points moved into frame j  -  frame-j depth at that pixel| (mm);
                  measures temporal/multi-view consistency, not absolute accuracy
  vs_raft_mm      median absolute difference to RAFT stereo
  ms / frame      mean latency after warm-up (includes pre/post-processing and, for mono, scale alignment)

    python scripts/run_depth.py data/scenes/stereomis_P2_1_tools --methods sgbm raft dav2_base
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sr3d.data.scene import Scene
from sr3d.depth.mono import DepthAnythingV2, align_inverse_depth
from sr3d.depth.raft import RAFTStereo
from sr3d.depth.sgbm import SGBMStereo
from sr3d.warp import photometric_error, warp_to_reference

DEPTH_SCALE = 100_000  # uint16 units per metre -> 10 um resolution, 0.65 m range


def build(name):
    if name == "sgbm":
        return SGBMStereo(num_disparities=128)
    if name == "raft":
        return RAFTStereo()
    if name.startswith("dav2_"):
        return DepthAnythingV2(name.split("_")[1])
    raise ValueError(name)


def predict(method, frame, K, baseline, ref_depth=None):
    if isinstance(method, DepthAnythingV2):
        rel = method.relative_inverse_depth(frame.rgb)
        # scale/shift from the stereo reference on tissue pixels; mono alone has no metric scale
        mask = None if frame.tool_mask is None else ~frame.tool_mask
        return align_inverse_depth(rel, ref_depth, mask)[0]
    return method.depth(frame.rgb, frame.right, K[0, 0], baseline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--methods", nargs="+", default=["sgbm", "raft", "dav2_small", "dav2_base", "dav2_large"])
    ap.add_argument("--every", type=int, default=3, help="evaluate every n-th frame")
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--out", default="outputs/depth")
    ap.add_argument("--reuse", action="store_true", help="load saved predictions instead of recomputing")
    a = ap.parse_args()

    scene = Scene(a.scene)
    out = Path(a.out) / scene.meta.name
    K, b = scene.K, scene.meta.baseline_m
    frames = list(range(0, len(scene) - a.gap, a.every))

    # RAFT first: it is the reference for mono alignment and agreement
    names = ["raft"] + [m for m in a.methods if m != "raft"]
    preds = {m: {} for m in names}
    timing = {}
    previous = out / "summary.json"
    for name in names:
        if a.reuse and previous.exists():
            for p in sorted((out / name).glob("*.png")):
                preds[name][int(p.stem)] = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
            timing[name] = json.loads(previous.read_text())[name]["ms_per_frame"]
            continue
        method = build(name)
        (out / name).mkdir(parents=True, exist_ok=True)
        times = []
        for k, i in enumerate(frames + [f + a.gap for f in frames]):
            if i in preds[name]:
                continue
            f = scene[i]
            ref = preds["raft"].get(i) if name != "raft" else None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            d = predict(method, f, K, b, ref)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if k > 0:
                times.append(time.perf_counter() - t0)
            preds[name][i] = d
            cv2.imwrite(str(out / name / f"{i:06d}.png"), np.clip(d * DEPTH_SCALE, 0, 65535).astype(np.uint16))
        timing[name] = 1000 * float(np.mean(times))
        del method
        torch.cuda.empty_cache()
        print(f"{name}: {len(preds[name])} frames, {timing[name]:.0f} ms/frame", flush=True)

    rows = {n: {"coverage": [], "warp_err": [], "warp_err_common": [], "geo_mm": [], "vs_raft_mm": []}
            for n in names}
    for i in frames:
        fi, fj = scene[i], scene[i + a.gap]
        tissue_i = ~fi.tool_mask if fi.tool_mask is not None else np.ones(fi.rgb.shape[:2], bool)
        tissue_j = ~fj.tool_mask if fj.tool_mask is not None else np.ones(fi.rgb.shape[:2], bool)
        common = tissue_i.copy()
        for n in names:
            common &= preds[n][i] > 0
        for n in names:
            d = preds[n][i]
            warped, valid = warp_to_reference(d, K, fi.T_wc, fj.T_wc, fj.rgb)
            # exclude target pixels that sample onto instruments in frame j
            uj, vj, zj = _lookup(d, K, fi.T_wc, fj.T_wc)
            tj = cv2.remap(tissue_j.astype(np.uint8), uj, vj, cv2.INTER_NEAREST) > 0
            valid &= tissue_i & tj
            dj = cv2.remap(preds[n][i + a.gap], uj, vj, cv2.INTER_NEAREST)
            geo = valid & (dj > 0)
            r = rows[n]
            r["coverage"].append(((d > 0) & tissue_i).sum() / tissue_i.sum())
            r["warp_err"].append(photometric_error(fi.rgb, warped, valid))
            r["warp_err_common"].append(photometric_error(fi.rgb, warped, valid & common))
            r["geo_mm"].append(1000 * float(np.median(np.abs(zj - dj)[geo])) if geo.any() else np.nan)
            both = (d > 0) & (preds["raft"][i] > 0) & tissue_i
            r["vs_raft_mm"].append(1000 * np.median(np.abs(d - preds["raft"][i])[both]))

    summary = {n: {k: float(np.nanmean(v)) for k, v in rows[n].items()} | {"ms_per_frame": timing[n]} for n in names}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'method':12s} {'coverage':>8s} {'warp_err':>8s} {'common':>8s} {'geo':>8s} {'vs_raft':>8s} {'ms':>6s}")
    for n, s in summary.items():
        print(f"{n:12s} {s['coverage']:8.1%} {s['warp_err']:8.2f} {s['warp_err_common']:8.2f} "
              f"{s['geo_mm']:6.2f}mm {s['vs_raft_mm']:6.2f}mm {s['ms_per_frame']:6.0f}")
    figure(scene, preds, names, frames, out / "qualitative.png")
    print(f"-> {out}")


def _lookup(depth, K, T_i, T_j):
    """Pixel coordinates (u, v) and depth z in frame j for every pixel of frame i."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth.astype(np.float64)
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1).reshape(-1, 3)
    T = np.linalg.inv(T_j) @ T_i
    q = p @ T[:3, :3].T + T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uj = (K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2]).reshape(h, w).astype(np.float32)
        vj = (K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]).reshape(h, w).astype(np.float32)
    return uj, vj, q[:, 2].reshape(h, w).astype(np.float32)


def figure(scene, preds, names, frames, path):
    pick = [frames[len(frames) // 5], frames[len(frames) // 2], frames[4 * len(frames) // 5]]
    fig, axes = plt.subplots(len(pick), len(names) + 1, figsize=(2.9 * (len(names) + 1), 2.5 * len(pick)))
    for r, i in enumerate(pick):
        f = scene[i]
        axes[r, 0].imshow(f.rgb)
        ref = preds["raft"][i]
        lo, hi = np.percentile(ref[ref > 0], [2, 98])
        for c, n in enumerate(names, 1):
            d = np.where(preds[n][i] > 0, preds[n][i], np.nan) * 1000
            im = axes[r, c].imshow(d, cmap="turbo", vmin=lo * 1000, vmax=hi * 1000)
            if r == 0:
                axes[r, c].set_title(n)
        axes[r, 0].set_title("left image" if r == 0 else "")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.6, label="depth [mm]")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
