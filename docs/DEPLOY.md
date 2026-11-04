# Deployment: TensorRT and Docker (Stage 7)

```
python scripts/export_trt.py --out outputs/deploy                           # ONNX + TensorRT engines (fp16, fp32)
python scripts/benchmark_deploy.py data/scenes/stereomis_P2_1_tools         # accuracy parity + latency
ros2 launch launch/proximity.launch.py backend:=tensorrt                    # node on the engines (sr3d-ros env)
```

The two per-frame networks are exported with fixed 640×512 inputs (`sr3d/deploy/engines.py`):
SegFormer-B2 returns the full-resolution instrument probability; RAFT returns the final flow for the batched
(left→right, right→left) pair, so the left-right consistency check stays in the same Python code as the
PyTorch path. TensorRT 11 only builds strongly typed networks, so precision comes from the ONNX graph: the
FP16 graph is converted after export with ONNX's float16 converter, keeping I/O in float32.

**RAFT needs mixed precision.** Its pixel coordinates reach ~80 at 1/8 resolution, where float16 steps are
1/16, i.e. 0.5 px at full resolution: rounding of up to 0.25 px on a ~26 px disparity (9 cm depth, 4.16 mm
baseline), about 1% of depth (estimate, not measured). Element-wise coordinate
arithmetic and grid sampling therefore stay float32; convolutions and the correlation run in float16.

## Results (RTX 5060 Laptop, TensorRT 11.4, 60 frames of P2_1 for accuracy, 100 timed calls after 20 warm-up)

Latency per call includes pre- and post-processing (numpy in, numpy out), CUDA synchronised.

| Network | Backend | Latency median (P90) | Accuracy vs PyTorch |
|---|---|---|---|
| RAFT stereo (12 iterations) | PyTorch FP32 | 177 ms (181) | reference |
| | TensorRT FP32 | 114 ms (116) | **not usable**: median 0.32 mm, P99 3.9 mm, validity agreement 79.6% |
| | **TensorRT FP16 (mixed)** | **51 ms (54)** | median 0.046 mm, P99 0.39 mm, validity agreement 99.8% |
| SegFormer-B2 | PyTorch (bf16 autocast) | 25.7 ms (36.3) | IoU vs ground truth 0.9406 |
| | TensorRT FP32 | 19.2 ms (19.6) | IoU vs GT 0.9405, vs PyTorch 0.9997 |
| | **TensorRT FP16** | **8.5 ms (9.6)** | IoU vs GT 0.9405, vs PyTorch 0.9997 |
| SAM 2.1 large propagation | PyTorch | 176 ms / frame | offline video mode, not exported (not used by the node) |

IoU here is the mean per-frame IoU over frames with an instrument (60 frames, every 5th), a different
average from the pooled pixel IoU in `FUSION.md`.

**End to end** (`ProximityPipeline.step`: depth, segmentation, fusion, proximity; 140 consecutive frames):
PyTorch 244 ms median (P90 277), TensorRT FP16 102 ms (P90 131). In the ROS node: 243 → 100 ms compute,
258 → 115 ms latency, 3.7 → 8.3 Hz every-frame rate; 9.2 Hz sustained on the 30 Hz camera stream (`ROS.md`).
Proximity accuracy is unchanged on TensorRT: MAE 0.854 mm vs 0.850 mm (PyTorch) against the reference.

![Per-frame compute by stage](figures/deploy_latency.png)

**The RAFT FP32 engine is wrong, not slow.** The FP32 ONNX graph is correct: ONNX Runtime reproduces the
PyTorch flow to 0.0005 px (median). TensorRT 11.4's FP32 build of the same file gives P99 flow errors of
19 px, with and without TF32. The mixed-precision engine is within 0.12 px (P99). The cause was not
isolated; the FP32 engine is not used.

After TensorRT, the k-d tree proximity step (30 ms, CPU) is the second-largest cost after depth.

## Docker

`docker/Dockerfile` (ROS 2 Jazzy base, CUDA 12.8 PyTorch wheels, TensorRT 11.4, the node and launch files).
**It has not been built or run**: this machine has no Docker and no sudo. It mirrors the conda setup that
was run. TensorRT engines are specific to the GPU and TensorRT version, so they are built inside the
container (`scripts/export_trt.py`), not copied in.

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop (8 GB, compute 12.0), driver 592.82, WSL2 Ubuntu 26.04 |
| `sr3d` env | Python 3.11, PyTorch 2.11 + CUDA 12.8, Open3D 0.20, TensorRT 11.4 |
| `sr3d-ros` env | RoboStack ROS 2 Jazzy, Python 3.12, same PyTorch / Open3D / TensorRT |

Note: the Stage 3 depth benchmark logged 523 ms per RAFT frame (`outputs/depth/.../summary.json`); the
controlled measurement here (warm, synchronised, plugged in) is 177 ms. The earlier figure was a whole-run
average and was not a controlled latency measurement; its conditions (power state, I/O) were not recorded.
