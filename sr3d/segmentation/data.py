"""Instrument segmentation dataset built from StereoMIS tool masks.

Split by patient so test numbers measure generalisation to unseen anatomy, not memorised frames:
train = P2 sequences except P2_1, val = P1, test = P3. P2_1 is held out entirely because it is the
reconstruction scene used in later stages.

On disk (one folder per sequence, rectified left images at the pipeline resolution):

    <root>/<seq>/images/000123.png   RGB
    <root>/<seq>/masks/000123.png    uint8 {0, 255}, 255 = instrument
    <root>/<seq>/index.json          [{"frame": 123, "tool_frac": 0.08}, ...]
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SPLITS = {
    "train": ["P2_0", "P2_2", "P2_3", "P2_4", "P2_5", "P2_6", "P2_7", "P2_8"],
    "val": ["P1"],
    "test": ["P3"],
}
TOOL_MIN_FRAC = 0.005  # below this a frame counts as tool-free (stray label pixels at the border)
TROCAR_FRAC = 0.95  # above this the endoscope is inside the trocar: all metal, no anatomy, so dropped


def select_frames(fractions: list[tuple[int, float]], n_tool: int, n_empty: int) -> list[int]:
    """Evenly spaced frames in time: up to n_tool with an instrument, up to n_empty without.

    Frames above TROCAR_FRAC (camera passing through the port) are skipped entirely.

    Neighbouring frames at 50 fps are near-duplicates, so even temporal spacing gives the most
    distinct views per labelled image and stops long sequences from dominating.
    """
    def spread(frames, n):
        if len(frames) <= n:
            return frames
        return [frames[i] for i in np.linspace(0, len(frames) - 1, n).round().astype(int)]

    tool = [f for f, x in fractions if TOOL_MIN_FRAC <= x <= TROCAR_FRAC]
    empty = [f for f, x in fractions if x < TOOL_MIN_FRAC]
    return sorted(spread(tool, n_tool) + spread(empty, n_empty))


def export_sequence(seq_dir: str | Path, frames: list[int], dst: str | Path, out_width: int = 640) -> Path:
    """Rectify and write the chosen frames and their masks (see module docstring for the layout)."""
    from ..data.converters.stereomis import Rectifier, decode_mask, iter_video, load_calibration

    seq_dir, dst = Path(seq_dir), Path(dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    rect = Rectifier(load_calibration(seq_dir / "StereoCalibration.ini"), out_width)
    want, index = set(frames), []
    for idx, frame in iter_video(next(seq_dir.glob("*.mp4")), min(frames), max(frames) + 1, 1):
        if idx not in want:
            continue
        h = frame.shape[0] // 2
        left, _ = rect(frame[:h], frame[h:])
        raw = cv2.imread(str(seq_dir / "masks" / f"{idx + 1:06d}l.png"), cv2.IMREAD_UNCHANGED)
        mask = rect.resize_rectified_mask(decode_mask(raw).astype(np.uint8) * 255)
        cv2.imwrite(str(dst / "images" / f"{idx:06d}.png"), left)
        cv2.imwrite(str(dst / "masks" / f"{idx:06d}.png"), mask)
        index.append({"frame": idx, "tool_frac": float((mask > 127).mean())})
    (dst / "index.json").write_text(json.dumps(index))
    return dst


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_index(root: str | Path, seqs: list[str]) -> list[tuple[Path, Path]]:
    """(image path, mask path) pairs for the given sequences."""
    pairs = []
    for s in seqs:
        d = Path(root) / s
        for e in json.loads((d / "index.json").read_text()):
            pairs.append((d / "images" / f"{e['frame']:06d}.png", d / "masks" / f"{e['frame']:06d}.png"))
    return pairs


FLUOR_RED_FRAC = 0.25  # white-light frames have >= ~0.4 red-dominant pixels, NIR fluorescence ~0


def red_fraction(rgb: np.ndarray) -> float:
    """Fraction of red-dominant pixels; white-light tissue is red, fluorescence mode is grey + green."""
    r, g, b = [c.astype(np.int16) for c in cv2.split(rgb)]
    return float(((r > g + 20) & (r > b + 20)).mean())


def is_fluorescence(rgb: np.ndarray) -> bool:
    return red_fraction(rgb) < FLUOR_RED_FRAC


def simulate_fluorescence(rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Imitate da Vinci NIR/ICG fluorescence: grey anatomy with saturated green glow on perfused tissue.

    No training sequence has fluorescence frames, so this is the only way to show the model that
    grey does not mean metal. The glow is a smooth random field restricted to tissue (instruments do
    not take up the dye) and keeps the underlying texture.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = np.clip((gray - 128) * rng.uniform(0.8, 1.3) + 128 + rng.uniform(-20, 20), 0, 255)
    out = np.repeat(gray[..., None], 3, axis=2)
    if rng.random() < 0.8:
        h, w = gray.shape
        # organic blobs: blurred white noise, nudged towards bright smooth tissue so borders partly
        # follow organ outlines rather than being independent of the anatomy
        field = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), rng.uniform(20, 45))
        field = (field - field.mean()) / (field.std() + 1e-6)
        shade = cv2.GaussianBlur(gray, (0, 0), 8)
        field += rng.uniform(0, 1.0) * (shade - shade.mean()) / (shade.std() + 1e-6)
        thr = np.quantile(field, 1 - rng.uniform(0.15, 0.6))
        alpha = cv2.GaussianBlur(np.clip((field - thr) / 0.3, 0, 1), (0, 0), 3)
        tool = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
        alpha *= 1 - cv2.GaussianBlur(tool, (0, 0), 3)
        # glow brightness follows the tissue shading, as dye emission is lit by the same NIR source
        glow = np.clip(gray * rng.uniform(1.4, 2.5) + rng.uniform(0, 40), 0, 255)
        out[..., 1] = gray * (1 - alpha) + glow * alpha
        out[..., [0, 2]] *= (1 - alpha * rng.uniform(0.6, 0.95))[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


class SegDataset:
    """Torch-style dataset over exported pairs -> (float32 CHW normalised image, int64 HW mask).

    Training: random scale (0.75-1.5x), crop to `crop`, horizontal/vertical flip, then with
    probability `p_fluor` simulated fluorescence, otherwise hue/saturation jitter; brightness and
    contrast jitter always (endoscope light and white balance vary between patients).
    Evaluation (crop=None): full frame, no augmentation; `fluor_eval` applies a fixed per-frame
    simulated fluorescence so model selection can see that mode without touching the test set.
    """

    def __init__(self, pairs: list[tuple[Path, Path]], crop: int | None = None, seed: int = 0,
                 p_fluor: float = 0.0, fluor_eval: bool = False):
        self.pairs, self.crop, self.p_fluor, self.fluor_eval = pairs, crop, p_fluor, fluor_eval
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.pairs[i][0])), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(self.pairs[i][1]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if self.crop:
            img, mask = self._augment(img, mask)
        elif self.fluor_eval:
            img = simulate_fluorescence(img, mask, np.random.default_rng(i))
        x = (img.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1).copy(), mask.astype(np.int64)

    def _augment(self, img, mask):
        r, c = self.rng, self.crop
        s = max(r.uniform(0.75, 1.5), c / min(img.shape[:2]))
        size = (round(img.shape[1] * s), round(img.shape[0] * s))
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        y, x = r.integers(0, img.shape[0] - c + 1), r.integers(0, img.shape[1] - c + 1)
        img, mask = img[y:y + c, x:x + c], mask[y:y + c, x:x + c]
        if r.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if r.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        if r.random() < self.p_fluor:
            img = simulate_fluorescence(img, mask, r).astype(np.float32)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + r.uniform(-10, 10)) % 180
            hsv[..., 1] *= r.uniform(0.6, 1.4)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        img = np.clip((img - 128) * r.uniform(0.8, 1.2) + 128 + r.uniform(-25, 25), 0, 255).astype(np.uint8)
        return img, mask

    @staticmethod
    def worker_init(worker_id: int):
        """DataLoader worker_init_fn: give each worker its own augmentation stream."""
        import torch
        info = torch.utils.data.get_worker_info()
        info.dataset.rng = np.random.default_rng(info.seed % 2**32)
