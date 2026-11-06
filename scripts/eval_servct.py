"""Absolute stereo accuracy on SERV-CT (CT reference), comparable to Tables 2-3 of Edwards et al. 2022.

    python scripts/eval_servct.py data/raw/servct/extracted/SERV-CT
    python scripts/eval_servct.py data/raw/servct/extracted/SERV-CT --trt outputs/deploy/servct   # + engines

Methods: RAFT dense (no left-right check, comparable to the paper's dense networks), RAFT with the 1 px
left-right check (what the pipeline uses; reported with its coverage), SGBM with the same check, and
optionally the TensorRT RAFT engines built for 720x576 (`scripts/export_trt.py --height 576 --width 720`).
No parameter is tuned on SERV-CT.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sr3d.depth.raft import RAFTStereo
from sr3d.depth.sgbm import SGBMStereo
from sr3d.eval import servct

PAPER_BEST = {   # HSM(level1), the best method in the paper, mean RMSE 3D (mm) and Bad3 (%), noc / occ
    "Experiment_1": {"rmse_3d_mm": (3.18, 4.43), "bad3_pct": (8.34, 10.84), "rmse_disp_px": (1.75, 2.19)},
    "Experiment_2": {"rmse_3d_mm": (2.12, 2.98), "bad3_pct": (5.46, 9.22), "rmse_disp_px": (1.81, 2.73)},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--trt", help="directory with raft_fp16.engine / raft_fp32.engine for 720x576")
    ap.add_argument("--out", default="outputs/servct")
    a = ap.parse_args()
    samples = servct.load(a.root)
    print(f"{len(samples)} pairs; baselines {sorted({round(s.baseline_mm, 2) for s in samples})} mm, "
          f"reference depth {min(np.median(s.fx * s.baseline_mm / s.disparity[s.valid_noc]) for s in samples):.0f}-"
          f"{max(np.median(s.fx * s.baseline_mm / s.disparity[s.valid_noc]) for s in samples):.0f} mm (median per image)")

    methods = {"raft_dense": RAFTStereo(lr_check_px=0.0).disparity,
               "raft_lr1": RAFTStereo(lr_check_px=1.0).disparity,
               "sgbm_lr1": SGBMStereo(num_disparities=160).disparity}
    if a.trt:
        from sr3d.deploy.engines import TRTStereo
        for prec in ("fp16", "fp32"):
            p = Path(a.trt) / f"raft_{prec}.engine"
            if p.exists():
                methods[f"trt_{prec}_dense"] = TRTStereo(p, lr_check_px=0.0).disparity

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    for name, fn in methods.items():
        preds = [fn(s.left, s.right) for s in samples]
        for occ in (False, True):
            split = "occ" if occ else "noc"
            for s, p in zip(samples, preds):
                rows.append({"method": name, "split": split, "experiment": s.experiment, "image": s.name,
                             **servct.metrics(p, s, occ)})
            for exp in sorted({s.experiment for s in samples}):
                sel = [r for r in rows if r["method"] == name and r["split"] == split and r["experiment"] == exp]
                results.setdefault(name, {}).setdefault(exp, {})[split] = servct.summarise(sel)

    print(f"\n{'method':16s} {'experiment':13s} split  Bad3 %         RMSE disp px   RMSE 3D mm     median 3D mm  coverage")
    for name, exps in results.items():
        for exp, splits in exps.items():
            for split, m in splits.items():
                f = lambda k: f"{m[k]['mean']:6.2f} ±{m[k]['std']:5.2f}"
                print(f"{name:16s} {exp:13s} {split}    {f('bad3_pct')}  {f('rmse_disp_px')}  {f('rmse_3d_mm')}  "
                      f"{m['median_3d_mm']['mean']:6.2f}        {m['coverage']['mean']:.3f}")
    print("\npaper best (HSM level1), noc / occ:", json.dumps(PAPER_BEST))
    (out / "servct_results.json").write_text(json.dumps({"results": results, "paper_best": PAPER_BEST}, indent=1))
    (out / "servct_per_image.json").write_text(json.dumps(rows, indent=1))
    print(f"-> {out}/servct_results.json")


if __name__ == "__main__":
    main()
