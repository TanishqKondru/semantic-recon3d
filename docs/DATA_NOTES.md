# Data notes

Findings from validating each dataset before building on it. Each one would have silently corrupted
downstream results if left unchecked.

## StereoMIS

| Item | Finding | How it was verified |
|---|---|---|
| Video layout | 1280x2048 H.264, left image on top, right below; container reports 60.0 fps (P1, P3; P2_1 reports none) | Frame shape, `cv2.CAP_PROP_FPS` |
| Calibration | Baseline 4.16 mm, fx = 1034 px at full resolution (560 px at 640 wide) | `StereoCalibration.ini`, rectifier output |
| Poses | One TUM line per video frame (0-based), metres, da Vinci forward kinematics | Line count = frame count (8465) |
| Encoder jitter | Static camera shows +/-0.04 mm pose noise, so summed path length overstates motion | Trajectory plot of frames 0-1500 |
| **Video seeking** | `cv2.CAP_PROP_POS_FRAMES` lands early (requested 7349, decoded 7346) | Compared seek vs sequential decode, pixel error 0 at 7346. Converter now decodes sequentially |
| **Mask coordinates** | Masks are drawn on *rectified* half-resolution images; re-rectifying shifts them ~22 px | Instrument-overlap proxy IoU 0.34 -> 0.63, contour overlay |
| **Mask numbering** | 1-based: `000001l.png` belongs to video frame 0, so masks exist for even frames | Offset sweep, contour overlay, matches authors' `counter` in `video_dataset.py` |
| Mask polarity | 0 = instrument, 255 = tissue (opposite of EndoNeRF) | Visual check |
| **Mask encodings** | Two formats: 640x512 RGB {0, 255} (P1, P2_x) and 1280x1024 single-channel {0, 1} (P3). Instrument is always 0, so `< 128` marks every P3 pixel as tool. Use `decode_mask` | Pixel value survey per sequence |

### Mask sync, all sequences

`scripts/check_mask_sync.py` (proxy IoU between a low-saturation instrument detector and mask file frame+1+k). Every sequence peaks at k = 0, so the 1-based numbering holds everywhere.

| Seq | Proxy IoU at -2 / 0 / +2 | Note |
|---|---|---|
| P1 | 0.374 / **0.443** / 0.402 | |
| P2_0 | 0.453 / **0.557** / 0.474 | |
| P2_1 | 0.433 / **0.474** / 0.448 | control |
| P2_2 | 0.564 / **0.601** / 0.558 | |
| P2_3 | 0.536 / **0.543** / 0.516 | weak peak at stride 40; rerun with `--stride 10` |
| P2_4 | 0.182 / **0.203** / 0.180 | tool rarely visible (0-5% area); `--min-frac 0.01 --stride 10` |
| P2_5 | 0.284 / **0.294** / 0.271 | |
| P2_6 | 0.267 / **0.324** / 0.284 | |
| P2_7 | 0.202 / **0.233** / 0.201 | |
| P2_8 | 0.330 / **0.396** / 0.346 | |
| P3 | 0.448 / **0.496** / 0.449 | needs `decode_mask` ({0, 1} encoding) |

### Imaging modes and non-robotic objects

| Item | Finding | How it was verified |
|---|---|---|
| **P3 fluorescence** | P3 frames 0-8420 (365 of 815 exported) are NIR fluorescence mode (ICG): grey anatomy, dye glows saturated green. P1/P2 are white light (a few dozen borderline P2 frames, unchecked) | Fraction of red-dominant pixels (r > g+20 and r > b+20) < 0.25; contact sheets. Median saturation alone misses the green-glow frames |
| P1 plastic ring | Every tool-free P1 frame shows a teal plastic ring (unlabelled, not a robotic tool); the camera barely moves | Contact sheet of predictions on P1 empty frames |
| P3 white tubes | White plastic drains/suction tips are labelled background | Contact sheets around frames 7500-8500 |

Labels cover the robotic instruments only, so models that learn "unsaturated/shiny = instrument" flag the ring and tubes, and all of the green-glowing fluorescence anatomy.

### Useful segments in P2_1

From `scripts/scan_stereomis.py` (600-frame windows, jitter smoothed):

| Frames | Camera travel | Rotation | Frames with instrument |
|---|---|---|---|
| 0-1500 | static | 0 deg | ~0% |
| 6900-7500 | ~110 mm | ~13 deg | ~60% |
| 7650-8250 | ~120 mm | ~19 deg | 0% |

## EndoNeRF

Static camera, depth from STTR-light in arbitrary units, masks 255 = instrument.
Terms: non-commercial, no redistribution, cite Wang et al. MICCAI 2022.
