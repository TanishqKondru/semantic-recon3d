"""One command to check the whole system still works and still reproduces the reported numbers.

    python scripts/selfcheck.py              # ~2-4 min on the RTX 5060 Laptop (plugged in)
    python scripts/selfcheck.py --quick      # skip the 300-frame proximity replays

Each check prints PASS / FAIL / SKIP with the measured value and the threshold it is held to. The thresholds
are the reported results with a small margin, so a FAIL means something changed (code, weights, engines,
drivers), not that the method is bad. Exit code 1 if anything failed.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/scenes/stereomis_P2_1_tools"
SEG = ROOT / "outputs/segmentation/segformer_b2_fluor/best"
ENG = ROOT / "outputs/deploy"
EVAL = ROOT / "outputs/ros/P2_1_6900_7500_s2/proximity_eval.json"
status = {"PASS": 0, "FAIL": 0, "SKIP": 0}


def report(name, ok, detail):
    tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    status[tag] += 1
    print(f"[{tag}] {name:44s} {detail}", flush=True)


def power_note():
    try:
        st = [p.read_text().strip() for p in Path("/sys/class/power_supply").glob("*/status")]
        clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip()
        return f"power {'/'.join(st) or '?'}, GPU SM clock {clk} MHz"
    except Exception as e:      # informative only
        return f"power state unknown ({e})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print(power_note(), "- timings are only meaningful when plugged in (laptop GPU throttles on battery)\n")

    # 1. environment
    import torch
    report("CUDA GPU", torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else "none")
    for mod in ("tensorrt", "triton", "open3d"):
        try:
            m = __import__(mod)
            report(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:
            report(f"import {mod}", False, repr(e))

    # 2. unit tests
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT)
    report("unit tests (pytest)", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])

    # 3. data / weights present
    for p in (SCENE, SEG, ROOT / "outputs/fusion/P2_1_6900_7500_s2/depth_raft", EVAL):
        report(f"exists {p.relative_to(ROOT)}", p.exists(), "" if p.exists() else "missing: see docs/SETUP.md")
    if not (SCENE.exists() and SEG.exists()):
        return finish()

    from sr3d.data.scene import Scene
    from sr3d.depth.raft import RAFTStereo
    from sr3d.segmentation.model import InstrumentSegmenter
    scene = Scene(str(SCENE))
    fx, B = scene.K[0, 0], scene.meta.baseline_m
    frames = [scene[i] for i in range(0, len(scene), 30)]

    # 4. depth: TensorRT engines agree with PyTorch RAFT
    raft = RAFTStereo()
    ref = [raft.depth(f.rgb, f.right, fx, B) for f in frames]
    med_depth = float(np.median(np.concatenate([d[d > 0] for d in ref])))
    report("RAFT depth plausible (median 60-150 mm)", 0.06 < med_depth < 0.15, f"median depth {med_depth * 1e3:.1f} mm")
    from sr3d.deploy.engines import TRTSegmenter, TRTStereo
    for prec, tol in (("fp16", 0.1), ("fp32", 0.02)):
        p = ENG / f"raft_{prec}.engine"
        if not p.exists():
            report(f"TensorRT RAFT {prec} vs PyTorch", None, "no engine: python scripts/export_trt.py")
            continue
        eng = TRTStereo(p)
        d = [eng.depth(f.rgb, f.right, fx, B) for f in frames]
        diff = np.concatenate([np.abs(x - y)[(x > 0) & (y > 0)] for x, y in zip(d, ref)]) * 1e3
        agree = np.mean([((x > 0) == (y > 0)).mean() for x, y in zip(d, ref)])
        report(f"TensorRT RAFT {prec} vs PyTorch", np.median(diff) < tol and agree > 0.99,
               f"median |d depth| {np.median(diff):.3f} mm (< {tol}), valid agreement {agree:.4f} (> 0.99)")

    # 5. segmentation
    seg = InstrumentSegmenter(str(SEG))
    tool_frames = [f for f in frames if f.tool_mask.any()]
    iou = lambda x, y: (x & y).sum() / max((x | y).sum(), 1)
    m_pt = [seg.prob(f.rgb) > 0.5 for f in tool_frames]
    gt_iou = np.mean([iou(m, f.tool_mask) for m, f in zip(m_pt, tool_frames)])
    report("SegFormer IoU vs ground truth (P2_1)", gt_iou > 0.90,
           f"{gt_iou:.3f} (> 0.90; every 30th frame, the reported 0.94 is every 5th)")
    p = ENG / "segformer_fp16.engine"
    if p.exists():
        trt_seg = TRTSegmenter(p)
        m_trt = [trt_seg.prob(f.rgb) > 0.5 for f in tool_frames]
        v = np.mean([iou(x, y) for x, y in zip(m_trt, m_pt)])
        report("TensorRT SegFormer fp16 vs PyTorch", v > 0.99, f"IoU {v:.4f} (> 0.99)")
    del raft, seg
    torch.cuda.empty_cache()

    # 6. proximity: the online pipeline reproduces the Stage 6 numbers (cached depth / masks)
    if not a.quick and EVAL.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_proximity import DEPTH_SCALE, replay
        cache = ROOT / "outputs/fusion" / scene.meta.name
        n = len(scene)
        depth = [cv2.imread(str(cache / "depth_raft" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
                 / DEPTH_SCALE for i in range(n)]
        masks = [cv2.imread(str(cache / "prob_segformer" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for i in range(n)]
        rgb = [scene[i].rgb for i in range(n)]
        stored = json.loads(EVAL.read_text())["per_frame_m"]
        ref_d, saved = np.array(stored["ref"], float), np.array(stored["online_segformer"], float)
        out = {}
        for gpu in (False, True):
            t = time.perf_counter()
            res, _ = replay(scene.K, rgb, depth, masks, scene.poses, gpu_nn=gpu)
            out[gpu] = np.array([r.distance for r in res])
            ms = np.median([r.ms["proximity"] for r in res[10:]])
            tag = "GPU" if gpu else "CPU"
            same = np.array_equal(np.isnan(out[gpu]), np.isnan(saved))
            f = np.isfinite(saved) & np.isfinite(out[gpu])
            dev = np.abs(out[gpu][f] - saved[f]).max() * 1e6 if f.any() else np.inf
            report(f"proximity replay ({tag}) = stored Stage 6 run", same and dev < 1.0,
                   f"max deviation {dev:.3f} um (< 1 um), proximity step {ms:.1f} ms")
        sel = np.isfinite(ref_d) & np.isfinite(out[False])
        mae = np.abs(out[False][sel] - ref_d[sel]).mean() * 1e3
        report("proximity MAE vs reference (P2_1)", abs(mae - 0.85) < 0.05, f"{mae:.3f} mm (reported 0.85)")
    elif a.quick:
        report("proximity replays", None, "--quick")

    print("\nLive ROS 2 check (separate env, opens RViz):  conda activate sr3d-ros && "
          "ros2 launch launch/proximity.launch.py backend:=tensorrt rviz:=true")
    return finish()


def finish():
    print(f"\n{status['PASS']} passed, {status['FAIL']} failed, {status['SKIP']} skipped")
    return 1 if status["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
