# ROS 2 proximity node (Stage 6)

A ROS 2 Jazzy node that turns stereo video and the robot's kinematic camera pose into the number a virtual
fixture or safety monitor needs: the distance between each instrument and the tissue, every frame.

```
conda activate sr3d-ros                                              # RoboStack ROS 2 Jazzy + torch + TensorRT
ros2 launch launch/proximity.launch.py rate:=15.0 backend:=tensorrt rviz:=true
ros2 launch launch/proximity.launch.py lockstep:=true                # process every frame (no drops)
python scripts/eval_proximity.py data/scenes/stereomis_P2_1_tools    # accuracy (sr3d env, cached depth/masks)
python scripts/ros_latency_report.py outputs/ros/lockstep_*.csv outputs/ros/realtime15_*.csv
bash scripts/capture_rviz.sh 187 docs/figures/ros_rviz_raw.png       # needs a display with OpenGL (WSLg works)
```

## Interface

| Topic | Type | Content |
|---|---|---|
| `/stereo/{left,right}/image_rect` (in) | `sensor_msgs/Image` | rectified rgb8 pair, same stamp |
| `/stereo/left/pose` (in) | `geometry_msgs/PoseStamped` | kinematic camera pose in `world`, same stamp |
| `/stereo/right/camera_info` (in) | `sensor_msgs/CameraInfo` | K and baseline (P[0,3] = -fx·B) |
| `/sr3d/min_distance` | `sensor_msgs/Range` | closest instrument-tissue distance, +inf when no instrument |
| `/sr3d/tool_tips` | `geometry_msgs/PoseArray` | instrument tips (world) |
| `/sr3d/proximity` | `visualization_msgs/MarkerArray` | tips, closest-point lines (red < 2 mm, amber < 5 mm), distance text |
| `/sr3d/instrument_cloud` | `sensor_msgs/PointCloud2` | current instrument surface (world) |
| `/sr3d/semantic_cloud` | `sensor_msgs/PointCloud2` | current frame x y z rgb label confidence (camera frame) |
| `/sr3d/tissue_map` | `sensor_msgs/PointCloud2` | fused tissue surface with colour (world), on each map refresh |

`sr3d/ros/player.py` replays a converted scene the way a robot would publish it. The node
(`sr3d/ros/node.py`) is a thin wrapper around `sr3d/ros/pipeline.py` (pure Python, no ROS), so the offline
evaluation replays exactly the code the node runs. Image queues are one deep: when perception is slower than
the camera the node takes the newest frame and drops the rest instead of falling behind.

## Method

Per frame: RAFT stereo depth → SegFormer instrument mask → integrate tissue depth (instrument dilated by
5 px) into the TSDF → every 10th frame, rebuild a k-d tree from the map's surface points → per instrument
(connected component of the mask): the 3D tip and the distance to tissue.

* **Tissue = fused map + current frame.** The live frame cannot see the tissue under the tool, which is
  where the distance matters, so the map supplies it. The map only holds frames up to now (causal), so tissue
  the camera has just uncovered is missing; the current frame's own tissue pixels (outside a 12 px margin
  around the instrument) are queried as well.
* **Instrument points** are the stereo depth inside the instrument mask eroded by 1 px away from both the
  mask edge and holes in the depth map. Eroding only the mask edge left a 0.2 mm "contact" made of
  flying pixels at the rim of a depth hole.
* **Distance** is the minimum of the 5×5 median-filtered per-pixel distance image, so it needs a patch of
  the instrument, not one stray pixel. A coarse pass over every 16th point bounds the k-d search; the bounded
  result is exact (unit-tested) and cut the worst frame from 1.9 s to 46 ms.
* **Tip** = the mask pixel farthest from where the instrument enters the image (centroid of its border
  pixels), lifted with the median depth around it; reported only when it is at least 5 px inside the image.

## Evaluation without measured distances

StereoMIS has no measured tool-tissue distance. The reference isolates what the pipeline adds: same RAFT
depth, but ground-truth instrument masks and a tissue map fused from all 300 frames (future included). Hover
frames (reference ≥ 2 mm) and contact frames penalise opposite errors and are reported separately. Alarm
recall/precision: frames where the node and the reference are both below the threshold.

### Tuning on P1 (validation patient), testing on P2_1

The two margins (live-tissue dilation, instrument erosion) were tuned on P1 only. A first tuning window had
the instrument on the tissue almost throughout (reference median 0.19 mm), so it could not penalise reading a
hovering tool as touching; it chose 0 px / 1 px, which failed on P2_1 (row 2 below). The tuning set was then
rebuilt from four P1 windows (frames 8400, 10800, 12600, 16050; 292 hover and 884 contact frames; a fifth
window, 13800, was dropped for missing ground-truth masks) and the objective became the mean of hover and
contact MAE. Grid: dilation 0–16 px × erosion 1–5 px; chosen 12 px / 1 px (P1 objective 1.12 mm vs 1.17 mm
for the initial 5 / 3). P2_1 was therefore scored three times in total; all three are listed.

| P2_1, ROS node configuration | MAE | Hover / contact MAE | Bias | Alarm < 2 mm R / P |
|---|---|---|---|---|
| Initial margins 5 px / 3 px (before any tuning) | 0.84 mm | 0.81 / 0.86 mm | +0.32 mm | 0.80 / 0.90 |
| First P1 tuning, contact-only window: 0 / 1 | 2.69 mm | 6.05 / 0.45 mm | −2.58 mm | 1.00 / 0.64 |
| **Final: balanced P1 tuning, 12 / 1** | **0.85 mm** | **1.10 / 0.68 mm** | **−0.05 mm** | **0.85 / 0.87** |

On P2_1 the final margins trade hover accuracy for contact accuracy and alarm recall; on the balanced
objective they are slightly worse than the initial ones (0.89 vs 0.84 mm). The difference is within the
spread between P1 windows; the margins are not a strong lever.

### Ablations on P2_1 (final margins, 228 frames with a reference distance)

| Variant | MAE | Median | P90 | Bias | Hover / contact | Alarm < 2 mm R / P |
|---|---|---|---|---|---|---|
| **ROS node: SegFormer, causal map + live tissue** | **0.85** | **0.55** | **1.68** | **−0.05** | **1.10 / 0.68** | **0.85 / 0.87** |
| causal map only | 0.94 | 0.56 | 1.91 | +0.14 | 1.06 / 0.86 | 0.80 / 0.88 |
| map k-d tree rebuilt every frame | 0.80 | 0.52 | 1.44 | −0.17 | 1.13 / 0.58 | 0.95 / 0.85 |
| ground-truth masks (online) | 0.85 | 0.50 | 1.81 | −0.04 | 1.08 / 0.70 | 0.81 / 0.89 |
| map from all frames (offline, not causal) | 0.59 | 0.32 | 0.96 | −0.57 | 1.12 / 0.24 | 1.00 / 0.86 |
| current frame only, no map | 1.41 | 1.12 | 2.64 | +0.86 | 1.01 / 1.68 | 0.19 / 0.79 |

All distances in mm. Alarm < 5 mm: recall 0.99, precision 0.94 for the node. Tips: 87 of 90 reference tips
matched (within 50 px), median error 4.0 px / 1.16 mm against tips from ground-truth masks. No frame without
an instrument produced a distance.

* **The map matters where the tool touches tissue**: without it, contact error is 1.68 mm and only 19% of
  close approaches (< 2 mm) raise the alarm; with it, 0.68 mm and 85%.
* **Segmentation is not the bottleneck**: ground-truth masks give the same MAE (0.85 mm). On P2_1 the
  segmenter is in-domain (same animal as training).
* **Causality is**: the all-frame map reaches 0.24 mm on contact frames. The tissue under a tool that stays in
  place is only seen after it moves; an online system cannot have that.
* Rebuilding the tree every frame helps a little (0.80 mm) at extra cost; every 10th frame is kept.

![RViz and per-frame distance](figures/ros_proximity.png)

## Runtime (RTX 5060 Laptop, P2_1, 640×512)

Lockstep = the player waits for each answer (every frame processed); real-time = player at a fixed rate.
The scene takes every 2nd frame of a video whose container reports 60 fps, so 30 Hz is the camera rate.
Latency = image stamp to `/sr3d/min_distance` publish. Medians after 5 warm-up frames.

| Backend | Compute / frame | Lockstep latency (median / P90) | Real-time 30 Hz input | Real-time 15 Hz input |
|---|---|---|---|---|
| PyTorch | 243 ms | 258 / 291 ms (3.7 Hz) | 3.9 Hz, 13% of frames, latency 262 / 290 ms | 4.0 Hz, 26% of frames |
| TensorRT FP16 | 100 ms | 115 / 143 ms (8.3 Hz) | **9.2 Hz, 31% of frames, latency 120 / 147 ms** | 9.1 Hz, 61% of frames |

Stage medians (TensorRT): depth 56 ms, segmentation 9 ms, fusion 3 ms, proximity 30 ms.
The live node reproduces the offline replay: PyTorch distances within 0.042 mm on all 300 frames;
TensorRT median difference 0.047 mm, MAE vs reference 0.854 mm (PyTorch 0.850 mm).

## Engineering notes

* The camera pose is a synchronised `PoseStamped` topic, not a tf2 lookup: under GPU load the tf listener
  thread fell ~0.5 s behind, lookups failed and the node processed only 2.9 Hz at 15 Hz input. TF is still
  broadcast for RViz.
* ROS 2 comes from RoboStack (conda-forge) because the machine has no sudo; `sr3d-ros` is a separate env
  (Python 3.12) so the working `sr3d` env is untouched.
* RViz needs OpenGL: conda's Xvfb has no GLX, so the screenshot uses WSLg and grabs the RViz window directly
  (the rootless X server cannot read the root window).

## Limitations

* The reference is not ground truth: it shares the RAFT depth, so stereo errors on the instrument surface
  are not measured. A tracked tool or a phantom with known geometry would be needed for absolute accuracy.
* One test segment (P2_1, 300 frames) and one tuning patient (P1); P2_1 was scored three times (all reported).
* Hover error stays ~1.1 mm: the online estimate is limited by tissue the camera has not yet seen.
