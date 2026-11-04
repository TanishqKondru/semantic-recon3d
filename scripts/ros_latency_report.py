"""Summarise the node's per-frame timing logs (sr3d/ros/node.py `log` CSVs).

Lockstep runs give the per-frame compute and latency with no queueing; real-time runs (player at a fixed
rate, named `realtime<Hz>_<backend>.csv`) give the sustained processing rate and how many camera frames
were skipped. Rates are measured over
the span between the first and last processed frame, so model loading before the first frame is excluded.

    python scripts/ros_latency_report.py outputs/ros/lockstep_pytorch.csv outputs/ros/realtime15_pytorch.csv ...
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

STAGES = ("depth", "segmentation", "fusion", "proximity", "compute", "publish", "latency")


def summarise(path: Path, camera_hz: float | None, warmup: int = 5) -> dict:
    rows = list(csv.DictReader(path.open()))[warmup:]
    col = {k: np.array([float(r[f"{k}_ms"]) for r in rows]) for k in STAGES}
    stamps = np.array([float(r["stamp"]) for r in rows])
    out = {"frames": len(rows), **{f"{k}_median_ms": float(np.median(v)) for k, v in col.items()},
           "latency_p90_ms": float(np.quantile(col["latency"], 0.9)),
           "latency_max_ms": float(col["latency"].max())}
    span = stamps[-1] - stamps[0]
    out["rate_hz"] = (len(rows) - 1) / span if span > 0 else float("nan")
    if camera_hz:
        out["camera_hz"] = camera_hz
        out["processed_share"] = out["rate_hz"] / camera_hz
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="outputs/ros/latency_report.json")
    a = ap.parse_args()
    report = {}
    for log in map(Path, a.logs):
        m = re.match(r"realtime(\d+)", log.stem)
        s = summarise(log, float(m.group(1)) if m else None)
        report[log.stem] = s
        print(f"{log.stem:22s} {s['frames']:4d} frames  {s['rate_hz']:5.2f} Hz  compute {s['compute_median_ms']:6.1f} ms  "
              f"latency median {s['latency_median_ms']:6.1f} / P90 {s['latency_p90_ms']:6.1f} ms  "
              + "  ".join(f"{k} {s[f'{k}_median_ms']:.1f}" for k in STAGES[:4])
              + (f"  processed {100 * s['processed_share']:.0f}% of camera frames" if "processed_share" in s else ""))
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
