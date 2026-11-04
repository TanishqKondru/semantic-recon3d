"""Binary instrument segmentation metrics.

Pixel counts are accumulated over the whole split, so IoU/Dice are dataset-level (large and small
tools weigh by area). Tool-free frames are scored separately as a false-alarm rate, because IoU is
undefined on them and they are where a model hallucinates instruments on specular tissue.
"""
from __future__ import annotations

import numpy as np

TOOL_MIN_FRAC = 0.005  # same threshold as dataset selection


class SegMeter:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.frame_iou: list[float] = []
        self.empty_frames = self.empty_false_alarms = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred, gt: bool (H, W) instrument masks for one frame."""
        tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        if gt.mean() >= TOOL_MIN_FRAC:
            self.frame_iou.append(tp / max(tp + fp + fn, 1))
        else:
            self.empty_frames += 1
            self.empty_false_alarms += int(pred.mean() >= TOOL_MIN_FRAC)

    @property
    def n_frames(self) -> int:
        return len(self.frame_iou) + self.empty_frames

    def summary(self) -> dict:
        return {
            "iou": self.tp / max(self.tp + self.fp + self.fn, 1),
            "dice": 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1),
            "precision": self.tp / max(self.tp + self.fp, 1),
            "recall": self.tp / max(self.tp + self.fn, 1),
            "mean_frame_iou": float(np.mean(self.frame_iou)) if self.frame_iou else float("nan"),
            "empty_false_alarm_rate": self.empty_false_alarms / max(self.empty_frames, 1),
            "n_tool_frames": len(self.frame_iou),
            "n_empty_frames": self.empty_frames,
        }
