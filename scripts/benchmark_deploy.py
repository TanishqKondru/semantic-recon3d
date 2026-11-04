"""Accuracy parity and latency of the deployed networks: PyTorch vs TensorRT FP32 / FP16.

Accuracy on every `--stride`-th frame of the scene:
  depth   median |depth - PyTorch depth| (mm) on pixels valid in both, and agreement of the validity mask
  masks   instrument IoU against ground truth, and IoU against the PyTorch mask
Latency: median / P90 per call over `--runs` calls after `--warmup`, numpy in, numpy out (pre- and
post-processing included), CUDA synchronised. End to end: `ProximityPipeline.step` over the scene.
SAM 2 propagation is timed per frame on the same frames (PyTorch only; it is not exported).

    python scripts/benchmark_deploy.py data/scenes/stereomis_P2_1_tools
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sr3d.data.scene import Scene
from sr3d.deploy.engines import TRTSegmenter, TRTStereo
from sr3d.depth.raft import RAFTStereo
from sr3d.ros.pipeline import ProximityPipeline
from sr3d.segmentation.model import InstrumentSegmenter


def timed(fn, args_list, warmup, runs):
    for i in range(warmup):
        fn(*args_list[i % len(args_list)])
    torch.cuda.synchronize()
    ms = []
    for i in range(runs):
        t = time.perf_counter()
        fn(*args_list[i % len(args_list)])
        torch.cuda.synchronize()
        ms.append(1000 * (time.perf_counter() - t))
    return {"median_ms": float(np.median(ms)), "p90_ms": float(np.quantile(ms, 0.9))}


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--engines", default="outputs/deploy")
    ap.add_argument("--seg", default="outputs/segmentation/segformer_b2_fluor/best")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--e2e-frames", type=int, default=150)
    a = ap.parse_args()

    scene = Scene(a.scene)
    fx, B = scene.K[0, 0], scene.meta.baseline_m
    frames = [scene[i] for i in range(0, len(scene), a.stride)]
    eng = Path(a.engines)
    stereo = {"pytorch": RAFTStereo(), "trt_fp32": TRTStereo(eng / "raft_fp32.engine"),
              "trt_fp16": TRTStereo(eng / "raft_fp16.engine")}
    seg = {"pytorch": InstrumentSegmenter(a.seg), "trt_fp32": TRTSegmenter(eng / "segformer_fp32.engine"),
           "trt_fp16": TRTSegmenter(eng / "segformer_fp16.engine")}
    results = {"scene": scene.meta.name, "frames": len(frames), "gpu": torch.cuda.get_device_name(),
               "depth": {}, "segmentation": {}}

    ref_depth = [stereo["pytorch"].depth(f.rgb, f.right, fx, B) for f in frames]
    ref_mask = [seg["pytorch"](f.rgb) for f in frames]
    gt = [f.tool_mask for f in frames]
    for name, net in stereo.items():
        d = [net.depth(f.rgb, f.right, fx, B) for f in frames]
        diffs = np.concatenate([np.abs(x - r)[(x > 0) & (r > 0)] for x, r in zip(d, ref_depth)]) * 1e3
        agree = np.mean([((x > 0) == (r > 0)).mean() for x, r in zip(d, ref_depth)])
        t = timed(lambda f, net=net: net.depth(f.rgb, f.right, fx, B), [(f,) for f in frames], a.warmup, a.runs)
        results["depth"][name] = {"median_abs_diff_mm": float(np.median(diffs)), "p99_abs_diff_mm":
                                  float(np.quantile(diffs, 0.99)), "valid_agreement": float(agree), **t}
        print(f"depth {name:9s} {t['median_ms']:6.1f} ms (P90 {t['p90_ms']:.1f})  |d - torch| median "
              f"{np.median(diffs):.4f} mm, P99 {np.quantile(diffs, 0.99):.3f} mm, valid agreement {agree:.4f}",
              flush=True)
    for name, net in seg.items():
        m = [net.prob(f.rgb) > 0.5 for f in frames]
        has = [g.any() for g in gt]
        r = {"iou_gt": float(np.mean([iou(x, g) for x, g, h in zip(m, gt, has) if h])),
             "iou_vs_pytorch": float(np.mean([iou(x, y) for x, y in zip(m, ref_mask)])),
             **timed(lambda f, net=net: net.prob(f.rgb), [(f,) for f in frames], a.warmup, a.runs)}
        results["segmentation"][name] = r
        print(f"seg   {name:9s} {r['median_ms']:6.1f} ms (P90 {r['p90_ms']:.1f})  IoU vs GT {r['iou_gt']:.4f}, "
              f"vs PyTorch {r['iou_vs_pytorch']:.4f}", flush=True)

    # end to end: the node's per-frame step (depth, segmentation, fusion, proximity) over consecutive frames
    seq = [scene[i] for i in range(a.e2e_frames)]
    results["end_to_end"] = {}
    for name in ("pytorch", "trt_fp16"):
        pipe = ProximityPipeline(scene.K, lambda l, r, s=stereo[name]: s.depth(l, r, fx, B), seg[name].prob)
        rows = [pipe.step(f.rgb, f.right, f.T_wc).ms for f in seq]
        rows = rows[10:]    # warm-up: first map extractions and CUDA kernels
        r = {k: float(np.median([x[k] for x in rows])) for k in rows[0]}
        r["p90_total_ms"] = float(np.quantile([x["total"] for x in rows], 0.9))
        results["end_to_end"][name] = r
        print(f"e2e   {name:9s} " + "  ".join(f"{k} {v:.1f}" for k, v in r.items()), flush=True)
    del stereo, seg
    torch.cuda.empty_cache()

    from sr3d.segmentation.tracking import keyframes, load_sam2, track
    sam = load_sam2("large")
    clip = [f.rgb for f in seq[:100]]
    prompts = {i: ref for i, ref in zip(keyframes(len(clip), 50), [InstrumentSegmenter(a.seg)(clip[i])
                                                                   for i in keyframes(len(clip), 50)])}
    track(sam, clip[:10], {0: prompts[0]})    # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    track(sam, clip, prompts)
    torch.cuda.synchronize()
    results["sam2_large_ms_per_frame"] = 1000 * (time.perf_counter() - t) / len(clip)
    print(f"sam2  large     {results['sam2_large_ms_per_frame']:.1f} ms / frame (offline video propagation, "
          f"{len(clip)} frames)")

    out = Path("outputs/deploy") / f"benchmark_{scene.meta.name}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
