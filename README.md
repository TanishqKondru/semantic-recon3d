# semantic-recon3d

Stereo endoscopy → semantic 3D map of the surgical scene → live instrument-to-tissue distance in ROS 2,
on one laptop GPU (RTX 5060 Laptop, 8 GB). Data: StereoMIS (da Vinci Xi, in-vivo porcine).
Evaluation protocol: train on P2_0, P2_2–P2_8; tune on P1; test on P3 (segmentation) and the held-out
P2_1 segment (fusion, proximity). Every number below comes from a file under `outputs/` produced by the
scripts listed in the stage docs.

| Stage | Result | Doc |
|---|---|---|
| 1–2 Environment, data | Six silent dataset problems found and fixed before training: video seeking, mask coordinates, mask numbering, mask encodings, trocar frames, P3 fluorescence mode | [DATA_NOTES](docs/DATA_NOTES.md), [SETUP](docs/SETUP.md) |
| 3 Depth | RAFT stereo: 0.52 mm cross-frame consistency, 93% coverage; SGBM 0.69 mm; Depth Anything V2 large 0.94 mm | – |
| 3 Pose | Stereo VO 3.8 mm ATE (scale 1.01) vs COLMAP 6.9 mm and VGGT 5.6 mm (both after Sim(3), wrong metric scale) | – |
| 4 Segmentation | SegFormer-B2 P3 test IoU 0.805 (fluorescence 0.034 → 0.686 with simulated fluorescence) | [SEGMENTATION](docs/SEGMENTATION.md) |
| 4 Tracking | SAM 2 large: P3 flicker 0.061 → 0.044 (ground truth 0.041), IoU 0.801 → 0.788: stability, not accuracy | [SEGMENTATION](docs/SEGMENTATION.md) |
| 5 Fusion | Gaussian splatting 23.1 dB / 0.71 mm vs TSDF 19.1 dB / 0.82 mm on 37 held-out views; voted 3D labels IoU 0.98 vs 0.90 single-frame | [FUSION](docs/FUSION.md) |
| 6 ROS 2 node | Instrument-tissue distance MAE 0.85 mm vs a ground-truth-mask reference; alarm < 2 mm recall 0.85 / precision 0.87 | [ROS](docs/ROS.md) |
| Absolute accuracy | SERV-CT (CT ground truth): RAFT on par with the best published method (Bad3 7.0 / 6.5 %, disparity RMSE 1.70 / 1.83 px); as the pipeline runs it, 2.2 / 1.5 mm RMS 3D error. Synthetic scenes with known geometry: a hovering 8 mm shaft reads **+3.8 mm too far** (its underside is hidden from the camera); with the instrument shape prior (node default) **−0.1 mm, MAE 0.45 mm**, 2 mm alarm recall 0.06 → 0.88 | [ROS](docs/ROS.md#absolute-accuracy-measured-ground-truth) |
| 7 Deployment | TensorRT FP16 + GPU nearest-neighbour search: 87 ms per frame end to end (PyTorch 244 ms); the ROS node runs 10.3 Hz on the 30 Hz camera stream (107 ms latency), 8.2 Hz / 140 ms with the instrument shape prior (default). TensorRT FP32 bug found and worked around; Docker image built and tested (17/17 self-checks pass in the container) | [DEPLOY](docs/DEPLOY.md) |

![ROS node](docs/figures/ros_proximity.png)

## Layout

```
sr3d/data        canonical scene format, StereoMIS converter, poses
sr3d/depth       RAFT stereo, SGBM, Depth Anything V2
sr3d/pose        stereo VO, COLMAP, VGGT wrappers
sr3d/segmentation SegFormer, fluorescence simulation, SAM 2 tracking
sr3d/fusion      TSDF (Open3D VoxelBlockGrid), Gaussian splatting (gsplat), multi-view label voting
sr3d/ros         proximity core (no ROS), online pipeline, ROS 2 node and player, message conversions
sr3d/deploy      ONNX export, TensorRT engines and runners
scripts/         one script per experiment (selfcheck.py checks everything); launch/ ROS 2 launch + RViz; docker/
tests/           geometry, data conventions, metrics, augmentation, tracking, fusion, proximity
```

## Honest caveats

* P2_1 (fusion / proximity test segment) is the same animal as the segmentation training data.
* The proximity reference shares the stereo depth with the system under test; it is not measured ground truth.
  Against known geometry (synthetic), the visible-surface distance reads a hovering shaft 3.8 mm too far; a
  capsule prior with the instrument's specified radius fixes that (MAE 0.45 mm). On real P2_1 the shafts'
  curvature is not observable in the stereo depth, so the prior mostly runs in a cautious offset form whose
  false-alarm rate on real data is unknown. Instrument kinematics (not in StereoMIS) would settle it.
* The P2_1 proximity test was scored three times while the tuning set was fixed; all three are in `docs/ROS.md`.

## Check that it works

```
conda activate sr3d
python scripts/selfcheck.py            # 17 checks, PASS/FAIL against the reported numbers (~3 min; plug the laptop in)
python -m pytest -q tests              # unit tests only
```

`selfcheck.py` checks the GPU stack, the unit tests, that the TensorRT engines match PyTorch (depth to < 0.1 mm,
masks to IoU > 0.99), and that replaying the stored P2_1 run through the online pipeline (CPU and GPU
nearest-neighbour search) reproduces the per-frame distances to < 1 um and the 0.85 mm MAE. To see the live
system: `conda activate sr3d-ros && ros2 launch launch/proximity.launch.py backend:=tensorrt rviz:=true`
(the RViz window shows the fused tissue map, instrument tips and closest-point lines; red < 2 mm).

In Docker (GPU via the NVIDIA Container Toolkit; details in `docs/DEPLOY.md`):

```
docker build -f docker/Dockerfile -t sr3d-ros .
docker run --gpus all --rm --shm-size=1g --user $(id -u):$(id -g) -e HOME=/tmp -v sr3d-cache:/cache \
    -v $PWD/data:/ws/data:ro -v $PWD/outputs:/ws/outputs sr3d-ros python scripts/selfcheck.py
```

## License and data

Code: MIT (`LICENSE`). No dataset files are in this repository. Figures show StereoMIS frames
(Hayoz et al., non-commercial license). Evaluations also use SERV-CT (Edwards et al., MedIA 2022,
CC BY-NC-SA 4.0). See `docs/SETUP.md` for sources and terms.
