# Instrument segmentation (Stage 4)

SegFormer-B2 (MiT-B2 ImageNet encoder), binary instrument vs. background, trained on StereoMIS tool
masks with a patient split: train P2 (minus P2_1, kept for reconstruction), val P1, test P3.
Data quirks behind these choices are in [DATA_NOTES.md](DATA_NOTES.md).

```
python scripts/prepare_seg.py data/raw/stereomis_seq/StereoMIS_0_0_1 data/seg/stereomis
python scripts/train_seg.py data/seg/stereomis outputs/segmentation/segformer_b2_fluor
```

## Results (P3, unseen patient, evaluated once per run)

| Model | P3 all IoU / Dice | P3 white light IoU / Dice | P3 fluorescence IoU / Dice |
|---|---|---|---|
| Run 1: baseline augmentation | 0.243 / 0.391 | **0.883** / 0.938 | 0.034 / 0.065 |
| **Run 2: + simulated fluorescence, EMA** | **0.805 / 0.892** | 0.868 / 0.929 | **0.686 / 0.814** |

P3 has 450 white-light and 365 fluorescence (NIR/ICG) frames; no training or validation sequence has
fluorescence. Run 2 precision / recall on all of P3: 0.894 / 0.890. Model: `outputs/segmentation/segformer_b2_fluor/best`.

Validation (checkpoint selection): run 1 picked by P1 IoU (0.702, epoch 6). Run 2 picked by the mean of
P1 IoU and simulated-fluorescence P1 IoU (0.711 / 0.768, epoch 10). The first model scored 0.103 on
simulated-fluorescence P1, close to its real-fluorescence 0.034, so the simulation is a usable proxy.

## What made the difference

1. **Diagnosing P3 by imaging mode.** Overall IoU 0.243 looked like a broken model. Splitting by time,
   then by mode, showed white light at 0.883 and fluorescence at 0.034: the model had learned
   "grey/unsaturated = metal", which is true in white light and false when anatomy turns grey.
2. **Simulated fluorescence** (`simulate_fluorescence` in `sr3d/segmentation/data.py`, 30% of crops):
   grayscale plus a green glow that follows tissue shading, on tissue only (instruments do not take up
   the dye). Fluorescence IoU 0.034 -> 0.686, at a cost of 0.015 white-light IoU.
3. **EMA weights + stronger colour jitter.** Run 1 val IoU swung 0.42-0.70 between epochs; run 2
   stayed within about 0.06 until mild overfitting after epoch 10, which checkpoint selection absorbs.

## Temporal consistency with SAM 2

SegFormer masks prompt SAM 2.1 (all instruments as one object) at keyframes; SAM 2 propagates between
them (`sr3d/segmentation/tracking.py`, `scripts/run_tracking.py`). Clips are 150 frames (300 video frames, 5 s at the 60 fps the video container reports)
picked from ground-truth tool fractions only. Tuned on P1; P3 run once.

```
python scripts/run_tracking.py val outputs/segmentation/tracking_val_large.json --clips 12 --every 50 --sam large
python scripts/run_tracking.py test outputs/segmentation/tracking_test.json --clips 12 --every 50 --sam large
```

Validation (P1, 12 clips, IoU pooled over frames; ground-truth flicker 0.070):

| Method | IoU | Flicker |
|---|---|---|
| SegFormer, per frame | 0.750 | 0.088 |
| SAM 2 small, every 50 frames | 0.809 | 0.052 |
| SAM 2 small, confidence-gated (w = 25 / 50) | 0.776 / 0.770 | 0.051 / 0.053 |
| **SAM 2 large, every 50 frames** | **0.825** | 0.054 |

Earlier rounds: prompting the first frame only drifts (0.612 on 6 clips); prompting every 5-25 frames
re-injects SegFormer errors; forward-backward propagation and averaging with SegFormer did not help.

Test (P3, 12 clips: 5 fluorescence, 7 white light; ground-truth flicker 0.041):

| Method | All IoU / Dice | White light IoU | Fluorescence IoU | Flicker |
|---|---|---|---|---|
| SegFormer, per frame | **0.801** / 0.890 | **0.852** | 0.678 | 0.061 |
| SegFormer + SAM 2 large, every 50 | 0.788 / 0.881 | 0.824 | **0.700** | **0.044** |

The validation accuracy gain did not hold on the test patient. SAM 2 cuts flicker by 28% to near the
ground truth's and helps fluorescence, but one propagated wrong prompt (clip 12050: 0.763 -> 0.647)
outweighs the gains elsewhere; it wins 6 clips, loses 3, ties 3. Clip 6800 (tool ~1.5% of the image,
fluorescence) fails for both. Use per-frame SegFormer for single-frame accuracy and SAM 2 where stable
masks matter (fusion, distance). Figure: `docs/figures/tracking_test_clips.png`.

## Known limitations

- **Non-robotic objects.** Labels mark robotic instruments only. White plastic drains/suction tips (P3)
  and a teal plastic ring (P1) are predicted as instruments. This is most of the remaining error and
  why the tool-free false-alarm rate stays high (P1 0.81, P3 fluorescence frames 1.0, 15 frames).
- **Real fluorescence is still the weak mode** (0.686 vs 0.868). Real fluorescence frames for
  training would help most; any P3 frame used that way must leave the test set.
- P1 validation is a single patient with one static ring scene, so its numbers are noisy.
