"""Verify that StereoMIS kinematic poses are time-aligned with the video.

For frame pairs (i, j), warp frame j into frame i using stereo depth of i and the GT relative pose,
with the pose index shifted by `delta` video frames. A correct alignment minimises the photometric error
at delta = 0. Instrument pixels are excluded because tools move independently of the camera.

    python scripts/check_pose_sync.py data/scenes/stereomis_P2_1_tools data/raw/stereomis_seq/StereoMIS_0_0_1/P2_1
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr3d.data.converters.stereomis import Rectifier, load_calibration
from sr3d.data.poses import read_tum
from sr3d.data.scene import Scene
from sr3d.depth.raft import RAFTStereo
from sr3d.warp import photometric_error, warp_to_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("raw_seq")
    ap.add_argument("--gap", type=int, default=3, help="scene-frame gap between i and j")
    ap.add_argument("--pairs", type=int, default=25)
    ap.add_argument("--max-shift", type=int, default=16, help="pose shift range in video frames")
    ap.add_argument("--out", default="outputs/pose_sync")
    a = ap.parse_args()

    scene = Scene(a.scene)
    ex = scene.meta.extra
    start, stride = ex["start"], ex["stride"]
    raw = Path(a.raw_seq)
    _, T_raw = read_tum(raw / "groundtruth.txt")
    rect = Rectifier(load_calibration(raw / "StereoCalibration.ini"), scene.meta.width)
    pose = lambda video_idx: rect.pose_to_rectified(T_raw[video_idx])  # noqa: E731

    stereo = RAFTStereo()
    shifts = np.arange(-a.max_shift, a.max_shift + 1)
    idx = np.linspace(0, len(scene) - 1 - a.gap, a.pairs).round().astype(int)
    errs = np.full((len(idx), len(shifts)), np.nan)
    static = np.full(len(idx), np.nan)
    for r, i in enumerate(idx):
        fi, fj = scene[i], scene[i + a.gap]
        d = stereo.depth(fi.rgb, fi.right, scene.K[0, 0], scene.meta.baseline_m)
        if fi.tool_mask is not None:
            d[fi.tool_mask] = 0
        vi, vj = start + i * stride, start + (i + a.gap) * stride
        for c, s in enumerate(shifts):
            if 0 <= vi + s and vj + s < len(T_raw):
                warped, valid = warp_to_reference(d, scene.K, pose(vi + s), pose(vj + s), fj.rgb)
                if fj.tool_mask is not None:
                    valid &= ~fj.tool_mask  # approximate: ignore pixels covered by tools in j
                errs[r, c] = photometric_error(fi.rgb, warped, valid)
        warped, valid = warp_to_reference(d, scene.K, np.eye(4), np.eye(4), fj.rgb)
        static[r] = photometric_error(fi.rgb, warped, valid)

    # normalise per pair so pairs with large motion don't dominate
    rel = errs / np.nanmin(errs, axis=1, keepdims=True)
    curve = np.nanmedian(rel, axis=0)
    best = shifts[np.nanargmin(curve)]
    per_pair_best = shifts[np.nanargmin(errs, axis=1)]
    print(f"pairs: {len(idx)}, gap {a.gap * stride} video frames")
    print(f"photometric error with GT poses (shift 0): {np.nanmedian(errs[:, a.max_shift]):.2f}  "
          f"vs assuming no camera motion: {np.nanmedian(static):.2f}  (grey levels, lower is better)")
    print(f"best pose shift (median curve): {best:+d} video frames; per-pair best: "
          f"median {np.median(per_pair_best):+.0f}, IQR {np.percentile(per_pair_best, 25):+.0f}..{np.percentile(per_pair_best, 75):+.0f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(shifts, curve, marker="o")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("pose time shift [video frames]")
    ax.set_ylabel("relative photometric error")
    ax.set_title(f"Pose/video sync, {scene.meta.name}")
    fig.tight_layout()
    fig.savefig(out / f"{scene.meta.name}.png", dpi=110)
    print(f"plot -> {out / (scene.meta.name + '.png')}")


if __name__ == "__main__":
    main()
