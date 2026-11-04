"""Find StereoMIS segments with real camera motion and visible instruments.

    python scripts/scan_stereomis.py data/raw/stereomis_seq/StereoMIS_0_0_1/P2_1 --window 600
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

from sr3d.data.poses import read_tum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--window", type=int, default=600, help="segment length in video frames")
    ap.add_argument("--out", default="outputs/scan")
    a = ap.parse_args()
    seq = Path(a.seq)

    _, T = read_tum(seq / "groundtruth.txt")
    n = len(T)
    # smooth out encoder jitter before measuring motion
    t = uniform_filter1d(T[:, :3, 3], size=15, axis=0) * 1000
    rot = Rotation.from_matrix(T[:, :3, :3])
    ang = np.degrees((rot[0].inv() * rot).magnitude())

    tool = np.full(n, np.nan)
    for f in sorted((seq / "masks").glob("*l.png")):
        i = int(f.stem[:-1])
        if i < n:
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            tool[i] = (m < 128).mean()
    tool_filled = np.interp(np.arange(n), np.flatnonzero(~np.isnan(tool)), tool[~np.isnan(tool)])

    w = a.window
    rows = []
    for s in range(0, n - w, w // 4):
        seg = slice(s, s + w)
        travel = np.linalg.norm(np.ptp(t[seg], axis=0))
        rows.append((s, travel, np.ptp(ang[seg]), tool_filled[seg].mean(), (tool_filled[seg] > 0.02).mean()))
    rows = np.array(rows)

    print(f"{seq.name}: {n} frames, total translation extent {np.linalg.norm(np.ptp(t, axis=0)):.1f} mm, "
          f"rotation range {np.ptp(ang):.1f} deg, frames with tool >2%: {(tool_filled > 0.02).mean():.0%}")
    print(f"{'start':>6} {'travel_mm':>9} {'rot_deg':>7} {'tool_px':>7} {'tool_frames':>11}")
    score = rows[:, 1] / max(rows[:, 1].max(), 1e-9) + rows[:, 4]
    for r in rows[np.argsort(-score)][:8]:
        print(f"{int(r[0]):6d} {r[1]:9.1f} {r[2]:7.1f} {r[3]:7.1%} {r[4]:11.0%}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t - t[0]); ax[0].set_ylabel("translation [mm]"); ax[0].legend(list("xyz"))
    ax[1].plot(ang); ax[1].set_ylabel("rotation from start [deg]")
    ax[2].plot(tool_filled * 100); ax[2].set_ylabel("instrument pixels [%]"); ax[2].set_xlabel("video frame")
    fig.suptitle(f"StereoMIS {seq.name}")
    fig.tight_layout()
    fig.savefig(out / f"{seq.name}_scan.png", dpi=100)
    print(f"plot -> {out / (seq.name + '_scan.png')}")


if __name__ == "__main__":
    main()
