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
    ax.set_xlabel("P1 clip start frame (each clip 150 frames, 6 s)"), ax.set_ylabel("clip IoU"), ax.set_ylim(0, 1)
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
    ax.set_xlabel("P3 clip start frame (each clip 150 frames, 6 s)"), ax.set_ylabel("clip IoU"), ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(fontsize=8, ncol=3, loc="upper center"), ax.grid(axis="y", alpha=0.3)
    ax.set_title("Test patient P3, evaluated once: per-clip IoU", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "tracking_test_clips.png", dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    training_curves(out)
    tracking_clips(out)
    tracking_test(out)
    fluorescence_sim(out)
    qualitative(out)


if __name__ == "__main__":
    main()
