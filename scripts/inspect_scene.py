"""Sanity-check a converted scene: summary stats, a contact sheet and the camera trajectory.

    python scripts/inspect_scene.py data/scenes/stereomis_P2_1 --out outputs/inspect
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr3d.data.scene import Scene


def overlay(rgb, mask, color=(0, 255, 0), alpha=0.45):
    out = rgb.copy()
    out[mask] = (alpha * np.array(color) + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def contact_sheet(scene: Scene, n: int, path: Path):
    idx = np.linspace(0, len(scene) - 1, n).round().astype(int)
    cols = ["rgb"] + [k for k in ("right", "depth", "tool_mask") if scene.has(k)]
    fig, axes = plt.subplots(n, len(cols), figsize=(3.2 * len(cols), 2.6 * n), squeeze=False)
    for r, i in enumerate(idx):
        f = scene[i]
        for c, kind in enumerate(cols):
            ax = axes[r, c]
            if kind == "rgb":
                ax.imshow(f.rgb)
            elif kind == "right":
                ax.imshow(f.right)
            elif kind == "depth":
                d = np.where(f.depth > 0, f.depth, np.nan)
                ax.imshow(d, cmap="turbo")
            else:
                ax.imshow(overlay(f.rgb, f.tool_mask))
            ax.set_axis_off()
            if r == 0:
                ax.set_title(kind, fontsize=10)
        axes[r, 0].text(5, 20, f"#{i}", color="w", fontsize=9, bbox=dict(fc="k", alpha=0.5, lw=0))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def trajectory_plot(scene: Scene, path: Path):
    t = scene.poses[:, :3, 3] * 1000.0
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*t.T, lw=1)
    ax.scatter(*t[0], c="g", s=30, label="start")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    for k, name in enumerate("xyz"):
        ax2.plot(t[:, k] - t[0, k], label=name)
    ax2.set_xlabel("frame"); ax2.set_ylabel("displacement from start [mm]"); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--out", default="outputs/inspect")
    ap.add_argument("--rows", type=int, default=4)
    a = ap.parse_args()

    scene = Scene(a.scene)
    out = Path(a.out) / scene.meta.name
    out.mkdir(parents=True, exist_ok=True)
    m = scene.meta
    print(f"{m.name}: {len(scene)} frames {m.width}x{m.height} source={m.source} domain={m.domain}")
    print(f"  K fx={scene.K[0,0]:.1f} fy={scene.K[1,1]:.1f} cx={scene.K[0,2]:.1f} cy={scene.K[1,2]:.1f}")
    if m.baseline_m:
        print(f"  stereo baseline {m.baseline_m * 1000:.2f} mm")
    if scene.poses is not None:
        t = scene.poses[:, :3, 3]
        steps = np.linalg.norm(np.diff(t, axis=0), axis=1) * 1000
        extent = np.ptp(t, axis=0) * 1000
        print(f"  trajectory length {steps.sum():.1f} mm (includes encoder jitter), max step {steps.max():.2f} mm, "
              f"extent xyz {np.array2string(extent, precision=2)} mm")
    f0 = scene[0]
    if f0.depth is not None:
        v = f0.depth[f0.depth > 0]
        unit = "m" if m.depth_is_metric else "rel"
        print(f"  depth[0] valid {v.size / f0.depth.size:.0%}, range {v.min():.3f}-{v.max():.3f} {unit}")
    if f0.tool_mask is not None:
        print(f"  tool pixels in frame 0: {f0.tool_mask.mean():.1%}")

    contact_sheet(scene, min(a.rows, len(scene)), out / "contact_sheet.png")
    if scene.poses is not None:
        trajectory_plot(scene, out / "trajectory.png")
    print(f"  figures -> {out}")


if __name__ == "__main__":
    main()
