"""Render the report figures for the segmentation stage from saved checkpoints and logs.

    python scripts/make_figures.py docs/figures
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr3d.segmentation.data import is_fluorescence, simulate_fluorescence
from sr3d.segmentation.model import InstrumentSegmenter

SEG = Path("data/seg/stereomis")
RUNS = {"Run 1: baseline": "outputs/segmentation/segformer_b2",
        "Run 2: + sim. fluorescence, EMA": "outputs/segmentation/segformer_b2_fluor"}


def load(seq, frame):
    img = cv2.cvtColor(cv2.imread(str(SEG / seq / "images" / f"{frame:06d}.png")), cv2.COLOR_BGR2RGB)
    return img, cv2.imread(str(SEG / seq / "masks" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127


def error_map(img, pred, gt):
    """Green = correct instrument, red = false positive, blue = missed instrument."""
    out = (img * 0.55).astype(np.uint8)
    for m, c in ((pred & gt, (60, 220, 60)), (pred & ~gt, (240, 50, 50)), (~pred & gt, (60, 120, 255))):
        out[m] = (0.4 * out[m] + 0.6 * np.array(c)).astype(np.uint8)
    return out


def iou(pred, gt):
    u = (pred | gt).sum()
    return float("nan") if u == 0 else (pred & gt).sum() / u


def pick(seq, want_fluor, n, min_frac=0.04):
    idx = [r for r in json.loads((SEG / seq / "index.json").read_text()) if r["tool_frac"] >= min_frac]
    idx = [r["frame"] for r in idx if is_fluorescence(load(seq, r["frame"])[0]) == want_fluor]
    return [idx[i] for i in np.linspace(0, len(idx) - 1, n + 2).round().astype(int)[1:-1]]


def qualitative(out):
    rows = [("P3", f, "P3 white light") for f in pick("P3", False, 2)]
    rows += [("P3", f, "P3 fluorescence") for f in pick("P3", True, 2)]
    empty = [r["frame"] for r in json.loads((SEG / "P1" / "index.json").read_text()) if r["tool_frac"] < 0.005]
    rows.append(("P1", empty[len(empty) // 2], "P1 tool-free (plastic ring)"))
    models = {k: InstrumentSegmenter(f"{v}/best") for k, v in RUNS.items()}
    fig, ax = plt.subplots(len(rows), 3, figsize=(11, 2.75 * len(rows)))
    for r, (seq, f, label) in enumerate(rows):
        img, gt = load(seq, f)
        ax[r, 0].imshow(img)
        ax[r, 0].set_ylabel(f"{label}\nframe {f}", fontsize=9)
        for c, (name, m) in enumerate(models.items(), 1):
            pred = m(img)
            ax[r, c].imshow(error_map(img, pred, gt))
            score = iou(pred, gt)
            ax[r, c].set_title(f"{name}  IoU {score:.2f}" if score == score else f"{name}  (no tool in GT)", fontsize=9)
        ax[r, 0].set_title("input", fontsize=9)
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Instrument segmentation, unseen patient. Green = correct, red = false positive, blue = missed",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "seg_qualitative.png", dpi=110)


def fluorescence_sim(out):
    rng = np.random.default_rng(3)
    p1 = pick("P1", False, 1, 0.08)[0]
    img, gt = load("P1", p1)
    real, _ = load("P3", pick("P3", True, 3)[1])
    panels = [(img, "P1 white light (input)"), (simulate_fluorescence(img, gt, rng), "simulated fluorescence"),
              (simulate_fluorescence(img, gt, rng), "simulated fluorescence (another draw)"),
              (real, "real fluorescence, P3 (test only)")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))
    for a, (im, t) in zip(ax, panels):
        a.imshow(im), a.set_title(t, fontsize=10), a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "fluorescence_simulation.png", dpi=110)


def training_curves(out):
    pat = re.compile(r"epoch\s+(\d+).*?val IoU ([0-9.]+)(?:.*?sim-fluor IoU ([0-9.]+))?")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (name, run), color in zip(RUNS.items(), ("tab:gray", "tab:blue")):
        rows = [pat.search(l) for l in Path(f"{run}_train.log").read_text().splitlines()]
        rows = [m.groups() for m in rows if m]
        ep = [int(r[0]) + 1 for r in rows]
        ax.plot(ep, [float(r[1]) for r in rows], "o-", color=color, label=f"{name}: P1 IoU")
        if rows[0][2]:
            ax.plot(ep, [float(r[2]) for r in rows], "s--", color=color, alpha=0.6, label=f"{name}: simulated-fluor. P1 IoU")
        best = json.loads(Path(f"{run}/metrics.json").read_text())["best_epoch"] + 1
        ax.axvline(best, color=color, lw=0.8, ls=":")
    ax.set_xlabel("epoch"), ax.set_ylabel("validation IoU (P1)"), ax.set_ylim(0, 1), ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Validation IoU per epoch (dotted line = selected checkpoint)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "seg_training_curves.png", dpi=120)


def tracking_clips(out, log="outputs/segmentation/tracking_val_small_r2.log"):
    text = Path(log).read_text().replace("\r", "\n")
    rows = []
    for line in text.splitlines():
        if line.startswith("P1 "):
            vals = dict(re.findall(r"(\S+): ([0-9.]+)", line))
            rows.append((int(line.split()[1]), {k: float(v) for k, v in vals.items()}))
    methods = {"segformer": ("SegFormer, per frame", "tab:gray"),
               "sam2_every25": ("SAM 2, re-prompt every 25", "tab:orange"),
               "sam2_every50": ("SAM 2, re-prompt every 50", "tab:blue")}
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, (k, (label, color)) in enumerate(methods.items()):
        ax.bar(x + (i - 1) * 0.27, [r[1][k] for r in rows], 0.27, label=label, color=color)
    ax.set_xticks(x, [str(r[0]) for r in rows], fontsize=8)
    ax.set_xlabel("P1 clip start frame (each clip 150 frames, 5 s)"), ax.set_ylabel("clip IoU"), ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=3, loc="lower left"), ax.grid(axis="y", alpha=0.3)
    ax.set_title("SAM 2 propagation vs per-frame SegFormer on validation clips (P1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "tracking_val_clips.png", dpi=120)


def tracking_test(out, result="outputs/segmentation/tracking_test.json"):
    clips = json.loads(Path(result).read_text())["clips"]
    x = np.arange(len(clips))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fluor = [i for i, c in enumerate(clips) if c["mode"] == "fluorescence"]
    if fluor:
        ax.axvspan(min(fluor) - 0.5, max(fluor) + 0.5, color="tab:green", alpha=0.08, label="fluorescence clips")
    for i, (k, label, color) in enumerate((("segformer", "SegFormer, per frame", "tab:gray"),
                                           ("sam2_every50", "SegFormer + SAM 2 large, every 50", "tab:blue"))):
        ax.bar(x + (i - 0.5) * 0.38, [c[k]["iou"] for c in clips], 0.38, label=label, color=color)
    ax.set_xticks(x, [str(c["start"]) for c in clips], fontsize=8)
    ax.set_xlabel("P3 clip start frame (each clip 150 frames, 5 s)"), ax.set_ylabel("clip IoU"), ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(fontsize=8, ncol=3, loc="upper center"), ax.grid(axis="y", alpha=0.3)
    ax.set_title("Test patient P3, evaluated once: per-clip IoU", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "tracking_test_clips.png", dpi=120)


def fusion_heldout(out, scene_dir="data/scenes/stereomis_P2_1_tools", cache="outputs/fusion", frames=(100, 220)):
    """Held-out views rendered from the TSDF mesh and the Gaussian map (needs gsplat: run in the activated env)."""
    import open3d as o3d
    import torch

    from sr3d.data.scene import Scene
    from sr3d.eval.fusion_metrics import MeshRenderer
    from sr3d.fusion.gaussians import GaussianMap

    scene = Scene(scene_dir)
    c = Path(cache) / scene.meta.name
    K, w, h = scene.K, scene.meta.width, scene.meta.height
    tsdf = MeshRenderer(o3d.io.read_triangle_mesh(str(c / "tsdf_sam2.ply")))
    params = torch.load(c / "gsplat.pt")
    gs = GaussianMap(np.zeros((4, 3)), np.zeros((4, 3)), 1.0)
    gs.params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.cuda()) for k, v in params.items()})
    fig, ax = plt.subplots(len(frames), 4, figsize=(16, 3.3 * len(frames)))
    for r, i in enumerate(frames):
        f = scene[i]
        depth = cv2.imread(str(c / "depth_raft" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED) / 1e5
        _, t_rgb = tsdf.render(K, scene.poses[i], w, h)
        g_d, g_rgb = gs.render_numpy(K, scene.poses[i], w, h)
        err = np.where((g_d > 0) & (depth > 0) & ~f.tool_mask, np.abs(g_d - depth) * 1000, np.nan)
        panels = [(f.rgb, f"held-out frame {i}"), (np.clip(t_rgb, 0, 255).astype(np.uint8), "TSDF mesh"),
                  (np.clip(g_rgb, 0, 255).astype(np.uint8), "Gaussian splatting")]
        for k, (im, t) in enumerate(panels):
            ax[r, k].imshow(im), ax[r, k].set_title(t, fontsize=10)
        im = ax[r, 3].imshow(err, vmin=0, vmax=3, cmap="magma")
        ax[r, 3].set_title("Gaussian depth vs stereo (mm)", fontsize=10)
        fig.colorbar(im, ax=ax[r, 3], fraction=0.04)
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Held-out views of the fused tissue map (instruments are not part of the static map)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fusion_heldout.png", dpi=110)


def fusion_labels(out, scene_dir="data/scenes/stereomis_P2_1_tools", cache="outputs/fusion"):
    """The unmasked TSDF map from a pulled-back viewpoint, coloured by voted instrument probability."""
    import open3d as o3d

    from sr3d.data.scene import Scene
    from sr3d.eval.fusion_metrics import MeshRenderer
    from sr3d.fusion.labels import vote

    scene = Scene(scene_dir)
    c = Path(cache) / scene.meta.name
    K, w, h, n = scene.K, scene.meta.width, scene.meta.height, len(scene)
    mesh = o3d.io.read_triangle_mesh(str(c / "tsdf_none.ply"))
    pts = np.asarray(mesh.vertices)
    depth = [cv2.imread(str(c / "depth_raft" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED) / 1e5 for i in range(n)]
    prob = [cv2.imread(str(c / "prob_sam2" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) / 255.0 for i in range(n)]
    v = vote(pts, [(prob[i], depth[i], scene.poses[i]) for i in range(0, n, 2)], K)
    base = np.asarray(mesh.vertex_colors).copy()
    labelled = base * 0.6 + 0.4 * 0.5
    instr = np.nan_to_num(v.prob) > 0.5
    labelled[instr] = [1.0, 0.55, 0.1]
    # camera pulled back 40 mm along its optical axis from the middle frame, wider field of view
    T = scene.poses[n // 2].copy()
    T[:3, 3] -= T[:3, 2] * 0.04
    Kw = K.copy()
    Kw[0, 0] = Kw[1, 1] = K[0, 0] * 0.6
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, colors, title in ((ax[0], base, "Fused map, no instrument masking"),
                             (ax[1], labelled, f"Voted labels: instrument = orange ({instr.mean() * 100:.1f}% of vertices)")):
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        _, rgb = MeshRenderer(mesh).render(Kw, T, w, h)
        a.imshow(np.clip(rgb, 0, 255).astype(np.uint8)), a.set_title(title, fontsize=10)
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fusion_labels.png", dpi=110)

# validated categorical slots (dataviz reference palette, light): blue, orange, aqua, yellow; ink for reference
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED = "#0b0b0b", "#52514e"


def ros_proximity(out, rviz="docs/figures/ros_rviz_raw.png", result="outputs/ros/P2_1_6900_7500_s2/proximity_eval.json",
                  frame=186):
    """RViz render of the node's topics next to the distance it publishes, against the reference."""
    e = json.loads(Path(result).read_text())["per_frame_m"]
    t = np.arange(len(e["ref"]))
    img = plt.imread(rviz)[58:873, 364:1584]      # 3D view only (drop RViz panels)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a0.imshow(img), a0.axis("off")
    a0.set_title(f"RViz at frame {frame}: fused tissue map (colour), instrument points (cyan), tip (magenta)",
                 fontsize=9, color=INK)
    series = (("ref", "reference (GT masks, all-frame map)", INK, 1.6),
              ("online_segformer", "ROS node (causal map + live tissue)", SLOTS[0], 1.2),
              ("live_frame", "current frame only, no map", SLOTS[1], 1.0))
    for k, label, color, lw in series:
        a1.plot(t, np.array(e[k]) * 1e3, color=color, lw=lw, label=label)
    a1.axhline(2.0, color=MUTED, lw=0.8, ls="--")
    a1.text(2, 2.15, "2 mm alarm", fontsize=8, color=MUTED)
    a1.axvline(frame, color=MUTED, lw=0.8, ls=":")
    a1.set_ylim(0, 8), a1.set_xlim(0, len(t) - 1)
    a1.set_xlabel("frame (P2_1 test segment)", color=INK), a1.set_ylabel("instrument-tissue distance [mm]", color=INK)
    a1.grid(alpha=0.25), a1.legend(fontsize=8, loc="upper right")
    a1.set_title("Closest instrument-tissue distance per frame (gaps: no instrument)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "ros_proximity.png", dpi=120)


def deploy_latency(out, report="outputs/ros/latency_report.json"):
    """Per-frame compute of the ROS node, PyTorch vs TensorRT FP16, stacked by stage (lockstep runs)."""
    r = json.loads(Path(report).read_text())
    rows = (("PyTorch", r["lockstep_pytorch"]), ("TensorRT FP16", r["lockstep_tensorrt"]))
    stages = (("depth", "stereo depth (RAFT)"), ("segmentation", "segmentation (SegFormer)"),
              ("fusion", "TSDF fusion"), ("proximity", "proximity (k-d tree)"))
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for y, (name, s) in enumerate(rows):
        left = 0.0
        for (k, label), color in zip(stages, SLOTS):
            v = s[f"{k}_median_ms"]
            ax.barh(y, v, left=left, height=0.5, color=color, edgecolor="#fcfcfb", linewidth=2,
                    label=label if y == 0 else None)
            if v > 12:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=8, color=INK)
            else:      # too narrow to hold its value: label it above the bar
                ax.text(left + v / 2, y - 0.3, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
            left += v
        ax.text(left + 4, y, f"{s['compute_median_ms']:.0f} ms compute, {s['latency_median_ms']:.0f} ms latency "
                f"(every frame: {s['rate_hz']:.1f} Hz)", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.invert_yaxis(), ax.set_xlim(0, 420), ax.set_xlabel("median per-frame time [ms], RTX 5060 Laptop", color=INK)
    ax.grid(axis="x", alpha=0.25), ax.legend(fontsize=8, ncol=4, loc="lower right", bbox_to_anchor=(1, 1.02))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "deploy_latency.png", dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*", help="figure function names to run (default: all)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = [training_curves, tracking_clips, tracking_test, fluorescence_sim, qualitative, fusion_heldout,
               fusion_labels, ros_proximity, deploy_latency]
    for fn in figures:
        if not a.only or fn.__name__ in a.only:
            fn(out)


if __name__ == "__main__":
    main()
