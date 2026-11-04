"""Verify StereoMIS mask numbering per sequence with an offset sweep.

Instruments are grey metal on red tissue, so a low-saturation threshold is a crude but
label-independent instrument detector. For frames where the tool is moving, the proxy IoU
between that detector and mask file (frame + 1 + k) should peak at k = 0 if the 1-based
numbering holds.

    python scripts/check_mask_sync.py data/raw/stereomis_seq/StereoMIS_0_0_1 --seqs P1 P2_6 P3
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from sr3d.data.converters.stereomis import Rectifier, decode_mask, iter_video, load_calibration


def read_mask(mask_dir: Path, file_no: int):
    p = mask_dir / f"{file_no:06d}l.png"
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None
    return None if m is None else decode_mask(m)


def pick_frames(mask_dir: Path, n_frames: int, n: int = 12, step: int = 2,
                min_frac: float = 0.03, stride: int = 40) -> list[int]:
    """Frames (0-based) where the tool covers min_frac-40% of the image and moves between neighbouring masks."""
    cands = []
    for f in range(0, n_frames - 20, stride):
        a, b = read_mask(mask_dir, f + 1), read_mask(mask_dir, f + 1 + 2 * step)
        if a is None or b is None or not min_frac < a.mean() < 0.4:
            continue
        cands.append((np.logical_xor(a, b).mean(), f))
    cands.sort(reverse=True)
    return sorted(f for _, f in cands[:n])


def sweep(seq_dir: Path, offsets=range(-6, 7), **pick) -> dict:
    mask_dir = seq_dir / "masks"
    n_frames = sum(1 for _ in open(seq_dir / "groundtruth.txt"))
    frames = pick_frames(mask_dir, n_frames, **pick)
    if not frames:
        raise SystemExit(f"{seq_dir.name}: no frames with a visible, moving tool")
    rect = Rectifier(load_calibration(seq_dir / "StereoCalibration.ini"), out_width=None)
    video = next(seq_dir.glob("*.mp4"))
    scores = {k: [] for k in offsets}
    want = set(frames)
    for idx, frame in iter_video(video, frames[0], frames[-1] + 1, 1):
        if idx not in want:
            continue
        left, _ = rect(frame[: frame.shape[0] // 2], frame[frame.shape[0] // 2:])
        hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
        for k in offsets:
            m = read_mask(mask_dir, idx + 1 + k)
            if m is None:
                continue
            m = rect.resize_rectified_mask(m.astype(np.uint8)).astype(bool)
            proxy = (hsv[..., 1] < 90) & (hsv[..., 2] > 40)
            scores[k].append((proxy & m).sum() / max((proxy | m).sum(), 1))
    curve = {k: float(np.mean(v)) for k, v in scores.items() if v}
    best = max(curve, key=curve.get)
    return {"frames": frames, "curve": curve, "best_offset": best}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--min-frac", type=float, default=0.03, help="min tool area fraction for a test frame")
    ap.add_argument("--stride", type=int, default=40, help="frame stride when searching test frames")
    a = ap.parse_args()
    for s in a.seqs:
        r = sweep(Path(a.root) / s, min_frac=a.min_frac, stride=a.stride)
        curve = " ".join(f"{k:+d}:{v:.3f}" for k, v in r["curve"].items())
        print(f"{s:5s} best_offset={r['best_offset']:+d}  {curve}", flush=True)


if __name__ == "__main__":
    main()
