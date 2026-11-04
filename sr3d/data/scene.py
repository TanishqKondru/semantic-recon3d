"""Canonical on-disk scene format shared by every domain.

Every dataset is converted once into this layout; the rest of the pipeline only reads it.

    <scene>/
      meta.json            name, domain, source, width, height, K, depth_scale, baseline_m
      images/000000.png    left / mono RGB
      right/000000.png     rectified right RGB (stereo datasets only)
      depth/000000.png     uint16 depth, metres = value / depth_scale (optional)
      masks/tool/000000.png  uint8 {0,255} instrument mask (optional)
      poses.txt            TUM format "t tx ty tz qx qy qz qw", camera-to-world (optional)

Conventions: OpenCV camera frame (x right, y down, z forward), metres, T_wc maps camera -> world.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .poses import read_tum, write_tum

SUBDIRS = {"rgb": "images", "right": "right", "depth": "depth", "tool_mask": "masks/tool"}


@dataclass
class Frame:
    index: int
    rgb: np.ndarray                      # (H, W, 3) uint8
    K: np.ndarray                        # (3, 3) float64
    T_wc: np.ndarray | None = None       # (4, 4) camera-to-world
    depth: np.ndarray | None = None      # (H, W) float32 metres, 0 = invalid
    tool_mask: np.ndarray | None = None  # (H, W) bool
    right: np.ndarray | None = None      # (H, W, 3) uint8, rectified
    timestamp: float = 0.0


@dataclass
class SceneMeta:
    name: str
    domain: str                          # medical | industrial | indoor
    source: str                          # originating dataset
    width: int
    height: int
    K: list[list[float]]
    depth_scale: float = 1000.0          # uint16 units per metre (1000 -> mm)
    depth_is_metric: bool = True         # False when depth is only relative (e.g. EndoNeRF)
    baseline_m: float | None = None      # stereo baseline after rectification
    extra: dict = field(default_factory=dict)


class Scene:
    """Lazy reader for a converted scene."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = SceneMeta(**json.loads((self.root / "meta.json").read_text()))
        self.K = np.asarray(self.meta.K, dtype=np.float64)
        self.names = sorted(p.stem for p in (self.root / "images").glob("*.png"))
        pose_file = self.root / "poses.txt"
        self.timestamps, self.poses = read_tum(pose_file) if pose_file.exists() else (None, None)
        if self.poses is not None and len(self.poses) != len(self.names):
            raise ValueError(f"{self.root}: {len(self.poses)} poses for {len(self.names)} images")

    def __len__(self) -> int:
        return len(self.names)

    def has(self, kind: str) -> bool:
        return (self.root / SUBDIRS[kind]).is_dir()

    def _read(self, kind: str, name: str, flags: int) -> np.ndarray | None:
        path = self.root / SUBDIRS[kind] / f"{name}.png"
        if not path.exists():
            return None
        img = cv2.imread(str(path), flags)
        if img is None:
            raise IOError(f"failed to read {path}")
        return img

    def __getitem__(self, i: int) -> Frame:
        if not 0 <= i < len(self):
            raise IndexError(i)
        name = self.names[i]
        rgb = cv2.cvtColor(self._read("rgb", name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        right = self._read("right", name, cv2.IMREAD_COLOR)
        depth = self._read("depth", name, cv2.IMREAD_UNCHANGED)
        mask = self._read("tool_mask", name, cv2.IMREAD_GRAYSCALE)
        return Frame(
            index=i,
            rgb=rgb,
            K=self.K,
            T_wc=None if self.poses is None else self.poses[i],
            depth=None if depth is None else depth.astype(np.float32) / self.meta.depth_scale,
            tool_mask=None if mask is None else mask > 127,
            right=None if right is None else cv2.cvtColor(right, cv2.COLOR_BGR2RGB),
            timestamp=float(self.timestamps[i]) if self.timestamps is not None else float(i),
        )

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class SceneWriter:
    """Incrementally writes frames into the canonical layout."""

    def __init__(self, root: str | Path, meta: SceneMeta, overwrite: bool = False):
        self.root = Path(root)
        if (self.root / "meta.json").exists() and not overwrite:
            raise FileExistsError(f"{self.root} already converted; pass overwrite=True")
        self.meta = meta
        self.count = 0
        self.poses: list[np.ndarray] = []
        self.stamps: list[float] = []
        self.root.mkdir(parents=True, exist_ok=True)

    def _write(self, kind: str, name: str, img: np.ndarray) -> None:
        d = self.root / SUBDIRS[kind]
        d.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(d / f"{name}.png"), img):
            raise IOError(f"failed to write {d / name}.png")

    def add(self, rgb, T_wc=None, depth=None, tool_mask=None, right=None, timestamp=None) -> None:
        h, w = rgb.shape[:2]
        if (w, h) != (self.meta.width, self.meta.height):
            raise ValueError(f"frame {w}x{h} does not match meta {self.meta.width}x{self.meta.height}")
        name = f"{self.count:06d}"
        self._write("rgb", name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if right is not None:
            self._write("right", name, cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
        if depth is not None:
            scaled = np.clip(np.nan_to_num(depth) * self.meta.depth_scale, 0, 65535)
            self._write("depth", name, np.round(scaled).astype(np.uint16))
        if tool_mask is not None:
            self._write("tool_mask", name, tool_mask.astype(np.uint8) * 255)
        if T_wc is not None:
            self.poses.append(np.asarray(T_wc, dtype=np.float64))
        if self.poses and len(self.poses) != self.count + 1:
            raise ValueError("poses must be given for every frame or none")
        self.stamps.append(float(self.count if timestamp is None else timestamp))
        self.count += 1

    def close(self) -> None:
        (self.root / "meta.json").write_text(json.dumps(self.meta.__dict__, indent=2))
        if self.poses:
            write_tum(self.root / "poses.txt", np.asarray(self.stamps), np.stack(self.poses))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
