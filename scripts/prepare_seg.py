"""Export the patient-split instrument segmentation dataset from StereoMIS.

    python scripts/prepare_seg.py data/raw/stereomis_seq/StereoMIS_0_0_1 data/seg/stereomis
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sr3d.data.converters.stereomis import decode_mask
from sr3d.segmentation.data import SPLITS, export_sequence, select_frames

# per-sequence caps: (frames with an instrument, tool-free frames)
BUDGET = {"train": (700, 120), "val": (500, 100), "test": (800, 100)}


def tool_fractions(seq_dir: Path) -> list[tuple[int, float]]:
    out = []
    for f in sorted((seq_dir / "masks").glob("*l.png")):
        m = decode_mask(cv2.imread(str(f), cv2.IMREAD_UNCHANGED))
        out.append((int(f.name[:6]) - 1, float(m.mean())))
    return out


def job(args):
    seq_dir, dst, n_tool, n_empty = args
    frames = select_frames(tool_fractions(seq_dir), n_tool, n_empty)
    export_sequence(seq_dir, frames, dst)
    index = json.loads((dst / "index.json").read_text())
    fr = np.array([e["tool_frac"] for e in index])
    return seq_dir.name, len(index), int((fr >= 0.005).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    jobs = [(Path(a.src) / s, Path(a.dst) / s, *BUDGET[split]) for split, seqs in SPLITS.items() for s in seqs]
    with ProcessPoolExecutor(len(jobs)) as ex:
        for name, n, n_tool in ex.map(job, jobs):
            print(f"{name:5s} {n:4d} frames, {n_tool:4d} with instrument", flush=True)


if __name__ == "__main__":
    main()
