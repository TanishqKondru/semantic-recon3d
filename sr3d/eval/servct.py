"""SERV-CT (Edwards et al., MedIA 2022): 16 ex-vivo porcine stereo-endoscope pairs with CT-derived reference
disparity, the only measured ground truth for absolute stereo accuracy in this project (StereoMIS has none).

Metric definitions follow the official toolkit (github.com/surgical-vision/servcttk) and Tables 2-3 of the
paper: per image, over reference-valid pixels where the method returns a disparity,
  Bad3 (% of pixels with |disparity error| > 3 px), RMSE disparity (px), and
  RMSE 3D distance (mm) between the points reprojected with Q from the estimated and reference disparity,
then mean (+/- std) over the images of an experiment. "noc" excludes occluded / non-overlapping pixels;
"occ" includes them (only pixels outside the reference surface are excluded). Coverage is the fraction of
reference-valid pixels with an estimate (the paper's methods are dense, coverage 1).

The reference itself is only accurate to ~2 px RMS disparity / ~2 mm depth (paper, section 2), so errors
below that level cannot be resolved by this dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# occlusion image colours (BGR, as cv2 reads them), from servcttk.iotools.parse_occlusion_image
OUTSIDE_SURFACE = (255, 0, 0)          # blue: no reference; always excluded
OCCLUDED = ((0, 255, 255), (0, 0, 255), (0, 255, 0))   # yellow non-overlap, red / green occluded


@dataclass
class Sample:
    name: str
    experiment: str
    left: np.ndarray          # RGB uint8, rectified
    right: np.ndarray
    disparity: np.ndarray     # float32 px, left-to-right
    valid_occ: np.ndarray     # bool, reference available
    valid_noc: np.ndarray     # bool, reference available and not occluded
    Q: np.ndarray             # (4, 4) reprojection matrix, mm
    fx: float
    baseline_mm: float


def _mask(occ_bgr: np.ndarray, colours) -> np.ndarray:
    m = np.zeros(occ_bgr.shape[:2], bool)
    for c in colours:
        m |= np.all(occ_bgr == c, axis=-1)
    return m


def load(root: str | Path, reference: str = "Reference_CT") -> list[Sample]:
    """root = the extracted 'SERV-CT' folder (Experiment_1, Experiment_2)."""
    root = Path(root)
    out = []
    for exp in sorted(root.glob("Experiment_*")):
        for lp in sorted((exp / "Left_rectified").glob("*.png")):
            n = lp.stem
            ref = exp / reference
            occ = cv2.imread(str(ref / "OcclusionL" / f"{n}.png"))
            calib = json.loads((exp / "Rectified_calibration" / f"{n}.json").read_text())
            P2 = np.array(calib["P2"]["data"]).reshape(3, 4)
            Q = np.array(calib["Q"]["data"]).reshape(4, 4)
            outside = _mask(occ, [OUTSIDE_SURFACE])
            out.append(Sample(
                name=n, experiment=exp.name,
                left=cv2.cvtColor(cv2.imread(str(lp)), cv2.COLOR_BGR2RGB),
                right=cv2.cvtColor(cv2.imread(str(exp / "Right_rectified" / f"{n}.png")), cv2.COLOR_BGR2RGB),
                disparity=cv2.imread(str(ref / "Disparity" / f"{n}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256,
                valid_occ=~outside, valid_noc=~outside & ~_mask(occ, OCCLUDED), Q=Q,
                fx=float(P2[0, 0]), baseline_mm=float(-P2[0, 3] / P2[0, 0])))
    return out


def reproject(disp: np.ndarray, Q: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Pixels `sel` with disparity `disp` -> (N, 3) points in mm (cv2.reprojectImageTo3D convention)."""
    v, u = np.nonzero(sel)
    h = np.stack([u, v, disp[sel], np.ones_like(u)], 0).astype(np.float64)
    X = Q @ h
    return (X[:3] / X[3]).T


def metrics(pred: np.ndarray, s: Sample, occlusions: bool) -> dict:
    ref_valid = s.valid_occ if occlusions else s.valid_noc
    sel = ref_valid & (pred > 0)
    err = pred[sel] - s.disparity[sel]
    d3 = np.linalg.norm(reproject(pred, s.Q, sel) - reproject(s.disparity, s.Q, sel), axis=1)
    return {"bad3_pct": float(100 * np.mean(np.abs(err) > 3)), "rmse_disp_px": float(np.sqrt(np.mean(err ** 2))),
            "epe_px": float(np.mean(np.abs(err))), "bias_px": float(np.mean(err)),
            "rmse_3d_mm": float(np.sqrt(np.mean(d3 ** 2))), "median_3d_mm": float(np.median(d3)),
            "coverage": float(sel.sum() / ref_valid.sum())}


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in keys}
