"""EndoNeRF (Wang et al., MICCAI 2022) -> canonical scene.

Layout: images/*.png, depth/*.png (STTR-light stereo depth, arbitrary scale),
masks/*.png (255 = instrument), poses_bounds.npy (LLFF, N x 17).
The sample clips are recorded from a static endoscope, so depth is stored as relative.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..scene import SceneMeta, SceneWriter


def llff_to_opencv(poses_bounds: np.ndarray) -> tuple[np.ndarray, tuple[int, int, float]]:
    """LLFF rows (N,17) -> camera-to-world (N,4,4) in OpenCV axes, plus (H, W, focal).

    EndoNeRF stores its poses already in OpenCV-style axes (as consumed by EndoNeRF/EndoGaussian),
    so only the 3x4 block is used here; the 5th column carries H, W, focal.
    """
    p = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, f = p[0, :, 4]
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :4] = p[:, :, :4]
    return T, (int(H), int(W), float(f))


def _sorted(d: Path) -> list[Path]:
    return sorted(d.glob("*.png"))


def convert(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
    src, dst = Path(src), Path(dst)
    images, depths, masks = _sorted(src / "images"), _sorted(src / "depth"), _sorted(src / "masks")
    T, (H, W, f) = llff_to_opencv(np.load(src / "poses_bounds.npy"))
    n = len(images)
    if not (n == len(depths) == len(masks) == len(T)):
        raise ValueError(f"count mismatch: {n} images, {len(depths)} depth, {len(masks)} masks, {len(T)} poses")

    meta = SceneMeta(
        name=src.name, domain="medical", source="EndoNeRF", width=W, height=H,
        K=[[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]],
        depth_scale=1.0, depth_is_metric=False,
        extra={"note": "depth is raw STTR-light output in arbitrary units"},
    )
    with SceneWriter(dst, meta, overwrite=overwrite) as w:
        for i in range(n):
            rgb = cv2.cvtColor(cv2.imread(str(images[i])), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(depths[i]), cv2.IMREAD_UNCHANGED)
            if depth.ndim == 3:
                depth = depth[..., 0]
            mask = cv2.imread(str(masks[i]), cv2.IMREAD_GRAYSCALE) > 127
            w.add(rgb, T_wc=T[i], depth=depth.astype(np.float32), tool_mask=mask, timestamp=i)
    return dst
