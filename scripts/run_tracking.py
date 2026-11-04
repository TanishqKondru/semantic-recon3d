"""Compare per-frame SegFormer against SAM 2 propagation on contiguous StereoMIS clips.

Clips are picked from ground-truth tool fractions only (instrument visible through most of the
window), evenly spread over the sequence. Frames are taken every 2nd video frame (30 Hz at the 60 fps
the video container reports), which is exactly the set of frames with a mask. Tune `--every` on val (P1);
run test (P3) once.

    python scripts/run_tracking.py val outputs/segmentation/tracking_val.json --every 0 50 25 10 5
    python scripts/run_tracking.py test outputs/segmentation/tracking_test.json --every <best>
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from sr3d.data.converters.stereomis import Rectifier, decode_mask, iter_video, load_calibration
from sr3d.eval.seg_metrics import SegMeter
from sr3d.segmentation.data import SPLITS, TOOL_MIN_FRAC, is_fluorescence
from sr3d.segmentation.model import InstrumentSegmenter
from sr3d.segmentation.tracking import confident_keyframes, flicker, keyframes, load_sam2, track, uncertainty


def pick_clips(fractions: list[tuple[int, float]], n_clips: int, clip_len: int) -> list[int]:
    """Start frames of up to n_clips non-overlapping windows where >= 90% of masked frames show a tool."""
    fr = dict(fractions)
    starts = []
    for s in range(0, max(fr) - clip_len, 50):
        s -= s % 2
        vals = [fr.get(f) for f in range(s, s + clip_len, 2)]
        if None in vals:
            continue
        ok = np.mean([TOOL_MIN_FRAC <= v <= 0.95 for v in vals])
        if ok >= 0.9:
            starts.append(s)
    chosen = []
    for s in [starts[i] for i in np.linspace(0, len(starts) - 1, 4 * n_clips).round().astype(int)] if starts else []:
        if all(abs(s - c) >= clip_len for c in chosen):
            chosen.append(s)
    return [chosen[i] for i in np.linspace(0, len(chosen) - 1, min(n_clips, len(chosen))).round().astype(int)]


def load_clip(seq_dir: Path, start: int, clip_len: int, width: int = 640):
    rect = Rectifier(load_calibration(seq_dir / "StereoCalibration.ini"), width)
    frames, gts = [], []
    for idx, frame in iter_video(next(seq_dir.glob("*.mp4")), start, start + clip_len, 1):
        if idx % 2:
            continue
        h = frame.shape[0] // 2
        left, _ = rect(frame[:h], frame[h:])
        raw = cv2.imread(str(seq_dir / "masks" / f"{idx + 1:06d}l.png"), cv2.IMREAD_UNCHANGED)
        frames.append(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
        gts.append(rect.resize_rectified_mask(decode_mask(raw).astype(np.uint8)).astype(bool))
    return frames, gts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("out")
    ap.add_argument("--raw", default="data/raw/stereomis_seq/StereoMIS_0_0_1")
    ap.add_argument("--seg", default="outputs/segmentation/segformer_b2_fluor/best")
    ap.add_argument("--sam", default="small", choices=["small", "base_plus", "large"])
    ap.add_argument("--every", type=int, nargs="+", default=[0, 50, 25, 10, 5],
                    help="re-prompt interval in clip frames (every 2nd video frame); 0 = first frame only")
    ap.add_argument("--conf", type=int, nargs="*", default=[],
                    help="window sizes for confidence-gated prompting (best SegFormer frame per window)")
    ap.add_argument("--both-ways", action="store_true", help="also run forward+backward SAM 2 variants")
    ap.add_argument("--fuse", action="store_true", help="also run SAM 2 + SegFormer probability averages")
    ap.add_argument("--clips", type=int, default=6, help="clips per sequence")
    ap.add_argument("--clip-len", type=int, default=300, help="video frames per clip (60 fps container rate)")
    a = ap.parse_args()

    fractions = json.loads(Path("outputs/segmentation/stereomis_tool_fraction.json").read_text())
    seg, sam = InstrumentSegmenter(a.seg), load_sam2(a.sam)
    variants = [(k, bw) for k in a.every for bw in ([False, True] if a.both_ways else [False])]
    name = lambda k, bw: f"sam2_every{k}" + ("_2way" if bw else "")
    methods = ["segformer"] + [name(k, bw) for k, bw in variants]
    if a.fuse:
        methods += [name(k, bw) + "+seg" for k, bw in variants]
    methods += [f"sam2_conf{w}" for w in a.conf]
    modes = ["all", "white_light", "fluorescence"]
    meters = {m: {md: SegMeter() for md in modes} for m in methods}
    flick = {m: [] for m in methods + ["ground_truth"]}
    clips, t0 = [], time.time()
    for s in SPLITS[a.split]:
        for start in pick_clips(fractions[s], a.clips, a.clip_len):
            frames, gts = load_clip(Path(a.raw) / s, start, a.clip_len)
            mode = "fluorescence" if np.mean([is_fluorescence(f) for f in frames]) > 0.5 else "white_light"
            seg_prob = [seg.prob(f) for f in frames]
            per_frame = [p > 0.5 for p in seg_prob]
            preds = {"segformer": per_frame}
            for k, bw in variants:
                prob = track(sam, frames, {i: per_frame[i] for i in keyframes(len(frames), k)}, both_ways=bw)
                preds[name(k, bw)] = [p > 0.5 for p in prob]
                if a.fuse:
                    preds[name(k, bw) + "+seg"] = [(p + q) / 2 > 0.5 for p, q in zip(prob, seg_prob)]
            unc = [uncertainty(p) if m.any() else np.inf for p, m in zip(seg_prob, per_frame)]
            for w in a.conf:
                keys = confident_keyframes(unc, w)
                preds[f"sam2_conf{w}"] = [p > 0.5 for p in track(sam, frames, {i: per_frame[i] for i in keys})]
            row = {"seq": s, "start": start, "frames": len(frames), "mode": mode}
            for m, p in preds.items():
                cm = SegMeter()
                for pf, g in zip(p, gts):
                    for mt in (cm, meters[m]["all"], meters[m][mode]):
                        mt.update(pf, g)
                flick[m].append(flicker(p))
                row[m] = {"iou": cm.summary()["iou"], "flicker": flick[m][-1]}
            flick["ground_truth"].append(flicker(gts))
            row["ground_truth_flicker"] = flick["ground_truth"][-1]
            clips.append(row)
            print(f"{s} {start:5d} {mode:12s} " + "  ".join(f"{m}: {row[m]['iou']:.3f}" for m in methods) + f"  gt flicker {row['ground_truth_flicker']:.3f}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    summary = {m: {md: meters[m][md].summary() for md in modes if meters[m][md].n_frames} for m in methods}
    for m in methods:
        summary[m]["flicker"] = float(np.mean(flick[m]))
    result = {"split": a.split, "args": vars(a), "summary": summary,
              "ground_truth_flicker": float(np.mean(flick["ground_truth"])), "clips": clips}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    w = max(map(len, methods)) + 1
    print(f"\n{'method':{w}s} " + " ".join(f"{md + ' IoU':>18s}" for md in modes) + "   flicker"
          f"   (ground truth flicker {result['ground_truth_flicker']:.3f})")
    for m in methods:
        print(f"{m:{w}s} " + " ".join(f"{summary[m][md]['iou']:18.3f}" if md in summary[m] else f"{'-':>18s}"
                                     for md in modes) + f"   {summary[m]['flicker']:.3f}")


if __name__ == "__main__":
    main()
