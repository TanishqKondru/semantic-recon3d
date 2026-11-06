"""Summarise outputs/synthetic/synthetic_proximity.json (from eval_synthetic_proximity.py).

    python scripts/summarise_synthetic.py

Trials where the placed instrument intersects the tissue (true gap < 0.05 mm although >= 0.5 mm was requested:
the shaft dipped into a tissue fold) are reported separately as "contact": they resemble a tool pressed into
tissue, where the contact line itself is visible. Alarm analysis: recall / precision of
"measured < 2 mm" for "true gap < 2 mm", and the alarm threshold needed for full recall, chosen on the
first tissue frame and scored on the second (held out).
"""
import json
from pathlib import Path

import numpy as np

P = Path("outputs/synthetic/synthetic_proximity.json")


def stats(e):
    e = np.asarray(e)
    return f"mean {e.mean():+.2f}, MAE {np.abs(e).mean():.2f}, median {np.median(e):+.2f}, range [{e.min():+.2f}, {e.max():+.2f}] mm"


def alarm(rows, key, thr, true_thr=2.0):
    pos = np.array([r["true_gap_mm"] < true_thr for r in rows])
    hit = np.array([r[key] < thr for r in rows])
    rec = (pos & hit).sum() / max(pos.sum(), 1)
    prec = (pos & hit).sum() / max(hit.sum(), 1)
    return rec, prec, int(pos.sum())


def main():
    d = json.loads(P.read_text())
    rows = d["rows"]
    bad = [r for r in rows if r["true_gap_mm"] < 0.05]
    rows = [r for r in rows if r["true_gap_mm"] >= 0.05 and np.isfinite(r["gt_mask_mm"])]
    frames = sorted({r["frame"] for r in rows})
    out = {"trials_total": len(d["rows"]), "contact_trials": len(bad), "trials": len(rows), "frames": frames}
    print(f"{len(d['rows'])} trials: {len(rows)} hover (true gap > 0), {len(bad)} contact (intersecting), "
          f"tissue frames {frames}")
    if bad:
        for key in [k for k in bad[0] if k.endswith("_mm") and k.startswith(("gt_mask", "segformer"))]:
            m = [r[key] for r in bad]
            out[f"contact_{key}"] = {"median": float(np.median(m)), "max": float(np.max(m)),
                                     "alarm2_recall": float(np.mean(np.array(m) < 2.0))}
            print(f"contact {key:13s}: measured median {np.median(m):.2f}, max {np.max(m):.2f} mm, "
                  f"alarm < 2 mm fires in {np.mean(np.array(m) < 2.0) * 100:.0f}%")
    print(f"tool depth error (RAFT on the instrument, median |d|): "
          f"{np.median([r['tool_depth_abs_err_mm'] for r in rows]):.2f} mm;  SegFormer IoU on the rendered tool "
          f"{np.mean([r['seg_iou'] for r in rows]):.3f}")
    vis = [r["visible_gap_mm"] - r["true_gap_mm"] for r in rows]
    print(f"visible - true gap (unobservable underside):  {stats(vis)}")
    out["visible_minus_true"] = float(np.mean(vis))
    keys = [k for k in rows[0] if k.endswith("_mm") and k.startswith(("gt_mask", "segformer"))]
    for key in keys:
        e_true = [r[key] - r["true_gap_mm"] for r in rows]
        e_vis = [r[key] - r["visible_gap_mm"] for r in rows]
        print(f"{key:13s} vs true gap:    {stats(e_true)}")
        print(f"{key:13s} vs visible gap: {stats(e_vis)}")
        out[key] = {"bias_vs_true": float(np.mean(e_true)), "mae_vs_true": float(np.mean(np.abs(e_true))),
                    "bias_vs_visible": float(np.mean(e_vis)), "mae_vs_visible": float(np.mean(np.abs(e_vis)))}
        rec, prec, npos = alarm(rows, key, 2.0)
        out[key]["alarm2_recall"], out[key]["alarm2_precision"] = float(rec), float(prec)
        print(f"{key:13s} alarm 'measured < 2 mm' for true < 2 mm ({npos} trials): recall {rec:.2f}, precision {prec:.2f}")
        cal = [r for r in rows if r["frame"] == frames[0]]
        test = [r for r in rows if r["frame"] != frames[0]]
        need = max([r[key] for r in cal if r["true_gap_mm"] < 2.0], default=np.nan)
        thr = float(np.ceil(need * 10) / 10 + 0.1)
        rec, prec, npos = alarm(test, key, thr)
        out[key]["calibrated_threshold_mm"], out[key]["heldout_recall"], out[key]["heldout_precision"] = thr, float(rec), float(prec)
        print(f"{key:13s} threshold for full recall on frame {frames[0]}: {thr:.1f} mm -> held-out frame(s) "
              f"{frames[1:]}: recall {rec:.2f}, precision {prec:.2f} ({npos} positives)")
    for key in [k for k in rows[0] if k.endswith("_fitted")]:
        print(f"{key}: capsule fitted in {np.mean([r[key] > 0 for r in rows]) * 100:.0f}% of hover trials")
        out[key] = float(np.mean([r[key] > 0 for r in rows]))
    print("\nper requested gap (gt mask):")
    for g in sorted({r["gap_mm"] for r in rows}):
        sel = [r for r in rows if r["gap_mm"] == g]
        print(f"  {g:4.1f} mm: n {len(sel)}, true {np.mean([r['true_gap_mm'] for r in sel]):5.2f}, visible "
              f"{np.mean([r['visible_gap_mm'] for r in sel]):5.2f}, measured {np.mean([r['gt_mask_mm'] for r in sel]):5.2f} mm")
    (P.parent / "synthetic_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
