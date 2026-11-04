"""Pool `eval_proximity.py --sweep` results over several tuning scenes and pick the margins.

Hover frames (reference >= 2 mm) and contact frames penalise opposite errors: reading a hovering tool as
touching, or a touching tool as hovering. The objective is the mean of the pooled hover MAE and the pooled
contact MAE, so neither regime can dominate because it happens to have more frames.

    python scripts/pool_proximity_sweeps.py P1_16050_16650_s2 P1_8400_9000_s2 P1_10800_11400_s2 ...
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+", help="scene names (directories under --out)")
    ap.add_argument("--out", default="outputs/ros")
    ap.add_argument("--files", nargs="+", default=["proximity_sweep.json"], help="sweep files per scene to merge")
    a = ap.parse_args()

    pooled = defaultdict(lambda: defaultdict(float))
    for name in a.scenes:
        rows = {(r["live_dilate_px"], r["erode_px"]): r for f in a.files
                for r in json.loads((Path(a.out) / name / f).read_text())["rows"]}.values()
        for r in rows:
            key = (r["live_dilate_px"], r["erode_px"])
            w = r["frames_scored"]
            p = pooled[key]
            p["frames"] += w
            p["abs"] += r["mae_mm"] * w
            for part in ("hover", "contact"):
                k = r[f"frames_{part}"]
                p[f"n_{part}"] += k
                p[f"abs_{part}"] += (r[f"mae_{part}_mm"] * k) if k else 0.0
            p["signed"] += r["bias_mm"] * w
            p["alarm_hits"] += r["alarm2_recall"] * r["alarm2_ref_frames"]
            p["alarm_ref"] += r["alarm2_ref_frames"]
            p[f"mae_{name}"] = r["mae_mm"]

    print(f"{'dilate':>6} {'erode':>5} {'object':>6} {'hover':>6} {'contact':>7} {'MAE':>6} {'bias':>6} {'recall<2':>8}  "
          + "  ".join(n[:12] for n in a.scenes))
    table = []
    for (dil, ero), p in sorted(pooled.items()):
        hover, contact = p["abs_hover"] / p["n_hover"], p["abs_contact"] / p["n_contact"]
        row = {"live_dilate_px": dil, "erode_px": ero, "objective_mm": (hover + contact) / 2,
               "mae_hover_mm": hover, "mae_contact_mm": contact, "frames_hover": int(p["n_hover"]),
               "frames_contact": int(p["n_contact"]), "mae_mm": p["abs"] / p["frames"],
               "bias_mm": p["signed"] / p["frames"], "alarm2_recall": p["alarm_hits"] / max(p["alarm_ref"], 1),
               "per_scene_mae": {n: p[f"mae_{n}"] for n in a.scenes}}
        table.append(row)
        print(f"{dil:6d} {ero:5d} {row['objective_mm']:6.3f} {hover:6.3f} {contact:7.3f} {row['mae_mm']:6.3f} "
              f"{row['bias_mm']:+6.2f} {row['alarm2_recall']:8.2f}  "
              + "  ".join(f"{v:12.3f}" for v in row["per_scene_mae"].values()))
    best = min(table, key=lambda r: r["objective_mm"])
    print(f"chosen: live dilate {best['live_dilate_px']} px, erode {best['erode_px']} px (objective "
          f"{best['objective_mm']:.3f} mm; hover {best['mae_hover_mm']:.3f} over {best['frames_hover']} frames, "
          f"contact {best['mae_contact_mm']:.3f} over {best['frames_contact']})")
    (Path(a.out) / "proximity_tuning.json").write_text(json.dumps(
        {"scenes": a.scenes, "rows": table, "chosen": best}, indent=1))


if __name__ == "__main__":
    main()
