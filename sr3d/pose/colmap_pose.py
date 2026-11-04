"""Classical incremental structure-from-motion with COLMAP (Schoenberger & Frahm, CVPR 2016).

Uses the known calibration (PINHOLE, fixed), sequential matching for video, and masks out instruments
during feature extraction since tools move independently of the anatomy.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap


class COLMAPPose:
    name = "colmap"

    def __init__(self, workdir: str | Path, overlap: int = 10):
        self.workdir = Path(workdir)
        self.overlap = overlap

    def __call__(self, images: list[np.ndarray], K: np.ndarray,
                 tool_masks: list[np.ndarray | None] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (camera-to-world (N,4,4) with NaN for unregistered frames, registered flags (N,))."""
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        img_dir, mask_dir = self.workdir / "images", self.workdir / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir()
        names = [f"{i:06d}.png" for i in range(len(images))]
        for i, (n, im) in enumerate(zip(names, images)):
            cv2.imwrite(str(img_dir / n), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            m = tool_masks[i] if tool_masks is not None else None
            keep = np.ones(im.shape[:2], np.uint8) * 255 if m is None else (~m).astype(np.uint8) * 255
            cv2.imwrite(str(mask_dir / f"{n}.png"), keep)  # COLMAP: <image name>.png, 0 = ignore

        db = self.workdir / "database.db"
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "PINHOLE"
        reader.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
        reader.mask_path = str(mask_dir)
        pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = self.overlap
        pycolmap.match_sequential(db, pairing_options=pairing)

        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
        recs = pycolmap.incremental_mapping(db, img_dir, self.workdir / "sparse", options=opts)

        T = np.full((len(images), 4, 4), np.nan)
        if not recs:
            return T, np.zeros(len(images), bool)
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        for img in rec.images.values():
            if not img.has_pose:
                continue
            i = names.index(img.name)
            T_cw = np.eye(4)
            T_cw[:3, :4] = img.cam_from_world().matrix()
            T[i] = np.linalg.inv(T_cw)
        return T, ~np.isnan(T[:, 0, 0])
