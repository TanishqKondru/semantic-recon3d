"""Export RAFT stereo and SegFormer to ONNX and build TensorRT engines (fixed 640x512 input).

    python scripts/export_trt.py --out outputs/deploy
"""
import argparse
import time
from pathlib import Path

import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from transformers import SegformerForSemanticSegmentation

from sr3d.deploy.engines import RAFT_FP32_OPS, RAFTFlow, SegFormerProb, build_engine, export_onnx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/deploy")
    ap.add_argument("--seg", default="outputs/segmentation/segformer_b2_fluor/best")
    ap.add_argument("--raft-iters", type=int, default=12)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"])
    ap.add_argument("--only", nargs="+", default=["segformer", "raft"])
    a = ap.parse_args()
    out = Path(a.out)
    h, w = a.height, a.width

    nets = {
        "segformer": (lambda: SegFormerProb(SegformerForSemanticSegmentation.from_pretrained(a.seg)),
                      (torch.randn(1, 3, h, w),), ["x"], ()),
        "raft": (lambda: RAFTFlow(raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False), a.raft_iters),
                 (torch.rand(2, 3, h, w) * 2 - 1, torch.rand(2, 3, h, w) * 2 - 1), ["a", "b"], RAFT_FP32_OPS),
    }
    for name, (make, inputs, names, fp32_ops) in ((k, nets[k]) for k in a.only):
        for prec in a.precisions:
            t0 = time.time()
            onnx = export_onnx(make(), inputs, out / f"{name}_{prec}.onnx", names, prec == "fp16", fp32_ops)
            t1 = time.time()
            build_engine(onnx, out / f"{name}_{prec}.engine")
            print(f"{name} {prec}: ONNX {t1 - t0:.0f} s, engine {time.time() - t1:.0f} s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
