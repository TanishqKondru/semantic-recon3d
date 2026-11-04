# Semantic 3D fusion (Stage 5)

Fuses RAFT stereo depth, camera poses and instrument probabilities from the P2_1 reconstruction segment
(frames 6900-7500, every 2nd frame, 300 frames) into a static tissue map with per-point labels.
Every 8th frame (37) is held out and used only for scoring.

```
python scripts/prepare_fusion.py data/scenes/stereomis_P2_1_tools      # depth + SegFormer / SAM 2 probabilities
python scripts/run_fusion.py data/scenes/stereomis_P2_1_tools          # TSDF maps, held-out scores, label voting
python scripts/run_fusion.py data/scenes/stereomis_P2_1_tools --poses stereo_vo --masks sam2
conda activate sr3d && python scripts/run_gsplat.py data/scenes/stereomis_P2_1_tools   # needs gsplat's CUDA build env
```

Instrument masks on P2_1: SegFormer IoU 0.984, SegFormer + SAM 2 0.974. P2_1 is the same animal as the
segmentation training sequences (it was held out of training, but it is not an unseen patient).

## Scoring without ground-truth geometry

The map is rendered from each held-out camera (ray casting for the mesh, gsplat for Gaussians) and compared
with that frame on ground-truth tissue pixels: depth against the frame's own stereo depth (median, mm),
colour against its image (PSNR), and the share of tissue pixels the map covers. `ghost_pct` is the share of
map vertices that ground-truth masks, voted over all fusion frames, mark as instrument: moving-tool geometry
left in the static map. 3D label IoU compares per-vertex labels voted from predicted masks with those voted
from ground-truth masks.

## Results (37 held-out frames, da Vinci kinematic poses)

| Map | Depth error | PSNR | Coverage | Instrument ghosts | Size | Build |
|---|---|---|---|---|---|---|
| TSDF, no instrument masking | 0.82 mm | 19.0 dB | 99.0% | 3.55% | 283k vertices | 7 s |
| TSDF, ground-truth masks | 0.82 mm | 19.1 dB | 98.8% | 0.02% | 271k vertices | 1 s |
| TSDF, SegFormer masks | 0.82 mm | 19.1 dB | 98.8% | 0.02% | 271k vertices | 1 s |
| TSDF, SegFormer + SAM 2 masks | 0.82 mm | 19.1 dB | 98.8% | 0.01% | 271k vertices | 1 s |
| **Gaussian splatting** (SAM 2 masks) | **0.71 mm** | **23.1 dB** | 100% | - | 654k Gaussians | 3.8 min |
| TSDF, stereo VO poses (vision only) | 1.29 mm | 19.7 dB | 98.7% | 0.00% | 303k vertices | 1 s |

TSDF: 0.5 mm voxels, truncation 5 voxels, surface kept where seen in at least 3 frames, instrument masks
dilated by 5 px. Gaussians: initialised from 150k TSDF vertices, 7000 steps, L1 + 0.2 D-SSIM + 0.05 x depth L1
(mm) on tissue pixels, gsplat default densification; renders in 30 ms.

**3D labels** (unmasked TSDF map, where instruments are present to label):

| Labels from | Multi-view vote | Last frame only |
|---|---|---|
| SegFormer | **0.982** | 0.899 |
| SegFormer + SAM 2 | 0.979 | 0.894 |

## Findings

- Masking instruments matters for geometry, not for the held-out scores: without it 3.6% of the map is
  instrument geometry smeared along the tool's path; any predicted mask removes nearly all of it.
- Gaussian splatting beats the TSDF by 4 dB PSNR and 0.11 mm depth. Its 100% coverage partly comes
  from extrapolating colour into tissue that was always hidden behind the tube.
- Voting over views lifts 3D label IoU by about 0.08 over using a single frame's label.
- Vision-only poses (stereo VO, ATE 4.4 mm over 300 frames) cost 0.47 mm of depth consistency: drift
  doubles surfaces (303k vs 271k vertices).
- The TSDF's colour is blurry: a 0.5 mm voxel spans 3-4 pixels at 80 mm, and it cannot represent the
  specular highlights of a light that moves with the camera.

Figures: `docs/figures/fusion_heldout.png`, `docs/figures/fusion_labels.png` (`scripts/make_figures.py`).

Open3D 0.20's legacy `ScalableTSDFVolume` returned empty meshes here, so fusion uses the tensor
`VoxelBlockGrid` on the GPU.
