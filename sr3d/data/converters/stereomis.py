"""StereoMIS (Hayoz et al., IJCARS 2023) -> canonical scene.

Each sequence folder holds a stereo .mp4 (left image on top, right below), StereoCalibration.ini
(OpenCV 8-coefficient model, right camera extrinsics R/T in mm), groundtruth.txt
(da Vinci forward-kinematics camera poses, TUM format in metres, one line per video frame) and
masks/<frame+1>l.png (already rectified, 0 = instrument). Two encodings exist: half-resolution
RGB {0, 255} (most sequences, every other frame; every frame in P2_6-P2_8) and full-resolution
grayscale {0, 1} (all of P3). The conventions were verified per sequence with
scripts/check_mask_sync.py and contour overlays: see docs/DATA_NOTES.md.

We rectify with OpenCV, optionally downscale, and rotate the GT poses into the rectified
left-camera frame so that depth from rectified stereo and the poses agree.
"""
from __future__ import annotations

import configparser
from pathlib import Path

import cv2
import numpy as np

from ..poses import read_tum
from ..scene import SceneMeta, SceneWriter


def load_calibration(ini: str | Path) -> dict:
    c = configparser.ConfigParser()
    c.read(ini)
    L, R = c["StereoLeft"], c["StereoRight"]

    def kmat(s):
        return np.array([[float(s["fc_x"]), 0, float(s["cc_x"])], [0, float(s["fc_y"]), float(s["cc_y"])], [0, 0, 1]])

    return {
        "size": (int(float(L["res_x"])), int(float(L["res_y"]))),
        "K1": kmat(L), "K2": kmat(R),
        "d1": np.array([float(L[f"kc_{i}"]) for i in range(8)]),
        "d2": np.array([float(R[f"kc_{i}"]) for i in range(8)]),
        "R": np.array([float(R[f"R_{i}"]) for i in range(9)]).reshape(3, 3),
        "T": np.array([float(R[f"T_{i}"]) for i in range(3)]),
    }


class Rectifier:
    def __init__(self, calib: dict, out_width: int | None = None):
        size = calib["size"]
        R1, R2, P1, P2, *_ = cv2.stereoRectify(
            calib["K1"], calib["d1"], calib["K2"], calib["d2"], size, calib["R"], calib["T"].reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.maps = [cv2.initUndistortRectifyMap(calib[f"K{i}"], calib[f"d{i}"], Ri, Pi, size, cv2.CV_32FC1)
                     for i, Ri, Pi in ((1, R1, P1), (2, R2, P2))]
        self.R1 = R1
        self.scale = 1.0 if out_width is None else out_width / size[0]
        self.size = (round(size[0] * self.scale), round(size[1] * self.scale))
        self.K = P1[:3, :3].copy()
        self.K[:2] *= self.scale
        # P2[0,3] = -fx * baseline, in the units of calib["T"] (mm)
        self.baseline_m = float(-P2[0, 3] / P2[0, 0]) / 1000.0

    def __call__(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = []
        for img, (mx, my) in zip((left, right), self.maps):
            r = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
            if self.scale != 1.0:
                r = cv2.resize(r, self.size, interpolation=cv2.INTER_AREA)
            out.append(r)
        return out[0], out[1]

    def resize_rectified_mask(self, mask: np.ndarray) -> np.ndarray:
        """StereoMIS masks are already in rectified coordinates; only match the output size."""
        if mask.shape[1::-1] == self.size:
            return mask
        return cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

    def pose_to_rectified(self, T_wc: np.ndarray) -> np.ndarray:
        """Rectification rotates the left camera by R1 (x_rect = R1 x_orig)."""
        T_rect_orig = np.eye(4)
        T_rect_orig[:3, :3] = self.R1
        return T_wc @ np.linalg.inv(T_rect_orig)


def decode_mask(raw: np.ndarray) -> np.ndarray:
    """Raw StereoMIS mask (either encoding) -> bool instrument mask.

    Tissue is 255 or 1 depending on the encoding, instrument is always 0, so `< 128` would mark
    every pixel of a {0, 1} mask as instrument.
    """
    if raw.ndim == 3:
        raw = raw[..., 0]
    return raw == 0


def iter_video(path: Path, start: int, stop: int, stride: int):
    """Yield (frame_index, BGR frame) for start <= index < stop.

    Decodes sequentially from frame 0: CAP_PROP_POS_FRAMES seeking on these H.264 files lands
    a few frames early (measured -3 at frame 7349), which would misalign images, poses and masks.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    for _ in range(start):
        if not cap.grab():
            raise IOError(f"{path} ends before frame {start}")
    idx = start
    while idx < stop:
        if (idx - start) % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
        elif not cap.grab():
            break
        idx += 1
    cap.release()


def convert(seq_dir: str | Path, dst: str | Path, start: int = 0, stop: int | None = None, stride: int = 1,
            out_width: int | None = 640, overwrite: bool = False) -> Path:
    seq_dir, dst = Path(seq_dir), Path(dst)
    videos = sorted(seq_dir.glob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"expected one .mp4 in {seq_dir}, found {len(videos)}")
    rect = Rectifier(load_calibration(seq_dir / "StereoCalibration.ini"), out_width)
    stamps, T_gt = read_tum(seq_dir / "groundtruth.txt")
    stop = len(T_gt) if stop is None else min(stop, len(T_gt))

    meta = SceneMeta(
        name=f"{seq_dir.name}_{start}_{stop}_s{stride}", domain="medical", source="StereoMIS",
        width=rect.size[0], height=rect.size[1], K=rect.K.tolist(), baseline_m=rect.baseline_m,
        extra={"sequence": seq_dir.name, "start": start, "stop": stop, "stride": stride,
               "pose_source": "da Vinci forward kinematics"},
    )
    mask_dir = seq_dir / "masks"
    with SceneWriter(dst, meta, overwrite=overwrite) as w:
        for idx, frame in iter_video(videos[0], start, stop, stride):
            h = frame.shape[0] // 2
            left, right = rect(frame[:h], frame[h:])
            mask = None
            # mask files are numbered with a 1-based frame counter (authors' preprocessing)
            mask_file = mask_dir / f"{idx + 1:06d}l.png"
            if mask_file.exists():
                raw = decode_mask(cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED))
                mask = rect.resize_rectified_mask(raw.astype(np.uint8)).astype(bool)
            w.add(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), right=cv2.cvtColor(right, cv2.COLOR_BGR2RGB),
                  T_wc=rect.pose_to_rectified(T_gt[idx]), tool_mask=mask, timestamp=stamps[idx])
    return dst
