"""Estimate camera trajectories with several methods and score them against da Vinci kinematics.

All methods see the same keyframes (every `--step`-th frame) so the numbers are comparable.

Scores (see sr3d/eval/pose_metrics.py):
  ate_rmse_mm     absolute trajectory error after Sim(3) alignment (scale-free methods get their scale fitted)
  ate_se3_mm      same with rigid SE(3) alignment only; meaningful only for metric methods (stereo VO)
  rpe_trans_mm    relative translation error between consecutive keyframes, using the fitted scale
  rpe_rot_deg     relative rotation error between consecutive keyframes
  scale           fitted Sim(3) scale (1.0 = already metric)
  registered      fraction of keyframes the method produced a pose for
  s_total         wall-clock time for the whole trajectory

    python scripts/run_pose.py data/scenes/stereomis_P2_1_tools --methods stereo_vo colmap vggt
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sr3d.data.poses import write_tum
from sr3d.data.scene import Scene
from sr3d.depth.sgbm import SGBMStereo
from sr3d.eval.pose_metrics import align_trajectory, ate_rmse, evaluate


def run(name, scene, frames, out):
    """-> (camera-to-world (N,4,4), registered mask (N,), extra info dict)."""
    K = scene.K
    if name == "stereo_vo":
        from sr3d.pose.stereo_vo import StereoVO
        sgbm = SGBMStereo(num_disparities=128)
        vo = StereoVO(lambda l, r: sgbm.depth(l, r, K[0, 0], scene.meta.baseline_m))
        T, inliers = vo(frames, K)
        return T, np.ones(len(frames), bool), {"median_inliers": float(np.median(inliers[1:])),
                                                "failed_steps": int((inliers[1:] == 0).sum())}
    if name == "colmap":
        from sr3d.pose.colmap_pose import COLMAPPose
        T, reg = COLMAPPose(out / "colmap_work")([f.rgb for f in frames], K, [f.tool_mask for f in frames])
        return T, reg, {}
    if name == "vggt":
        from sr3d.pose.vggt_pose import VGGTPose
        model = VGGTPose()
        torch.cuda.reset_peak_memory_stats()
        T, Kp = model([f.rgb for f in frames])
        info = {"fx_pred": float(np.median(Kp[:, 0, 0])), "fx_calib": float(K[0, 0]),
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
        del model
        torch.cuda.empty_cache()
        return T, np.ones(len(frames), bool), info
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--methods", nargs="+", default=["stereo_vo", "colmap", "vggt"])
    ap.add_argument("--step", type=int, default=6, help="use every n-th frame as a keyframe")
    ap.add_argument("--max-frames", type=int, default=50, help="VGGT memory bound on an 8 GB GPU")
    ap.add_argument("--out", default="outputs/pose")
    a = ap.parse_args()

    scene = Scene(a.scene)
    out = Path(a.out) / scene.meta.name
    out.mkdir(parents=True, exist_ok=True)
    idx = np.arange(0, len(scene), a.step)[: a.max_frames]
    frames = [scene[i] for i in idx]
    gt = scene.poses[idx]
    stamps = scene.timestamps[idx]
    write_tum(out / "gt.txt", stamps, gt)

    summary, aligned = {}, {}
    for name in a.methods:
        t0 = time.perf_counter()
        T, reg, info = run(name, scene, frames, out)
        dt = time.perf_counter() - t0
        write_tum(out / f"{name}.txt", stamps[reg], T[reg])
        m = evaluate(T[reg], gt[reg])
        m["ate_se3_mm"] = 1000 * ate_rmse(T[reg], gt[reg], with_scale=False)
        summary[name] = m | {"registered": float(reg.mean()), "s_total": dt} | info
        aligned[name] = (reg, align_trajectory(T[reg], gt[reg])[0])
        print(f"{name}: {reg.sum()}/{len(idx)} keyframes in {dt:.1f}s", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(idx)} keyframes, GT path {_path_mm(gt):.0f} mm")
    print(f"{'method':10s} {'ATE':>8s} {'ATE se3':>8s} {'RPE t':>8s} {'RPE r':>7s} {'scale':>7s} {'reg':>5s} {'time':>6s}")
    for n, s in summary.items():
        print(f"{n:10s} {s['ate_rmse_mm']:6.2f}mm {s['ate_se3_mm']:6.1f}mm {s['rpe_trans_mm']:6.2f}mm "
              f"{s['rpe_rot_deg']:6.2f}° {s['scale']:7.3f} {s['registered']:5.0%} {s['s_total']:5.1f}s")
    figure(gt, aligned, out / "trajectories.png")
    print(f"-> {out}")


def _path_mm(T):
    return 1000 * float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum())


def figure(gt, aligned, path):
    """Top and side views of the Sim(3)-aligned trajectories, in mm relative to the first GT pose."""
    o = gt[0, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (i, j, lab) in zip(axes, [(0, 1, ("x", "y")), (0, 2, ("x", "z"))]):
        g = 1000 * (gt[:, :3, 3] - o)
        ax.plot(g[:, i], g[:, j], "k-", lw=2.5, label="da Vinci kinematics")
        for name, (reg, T) in aligned.items():
            p = 1000 * (T[:, :3, 3] - o)
            ax.plot(p[:, i], p[:, j], "-", lw=1.5, label=name)
        ax.set_xlabel(f"{lab[0]} [mm]")
        ax.set_ylabel(f"{lab[1]} [mm]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Camera trajectories after Sim(3) alignment")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
