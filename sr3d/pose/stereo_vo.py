"""Metric frame-to-frame stereo visual odometry: stereo depth + ORB matches + PnP-RANSAC.

Unlike COLMAP and VGGT this recovers absolute scale from the stereo baseline, which is what a robot
controller needs. Instruments are masked out of feature detection since they move independently of
the anatomy. Errors accumulate (no loop closure / bundle adjustment), so drift grows with path length.
"""
from __future__ import annotations

import cv2
import numpy as np


class StereoVO:
    name = "stereo_vo"

    def __init__(self, depth_fn, n_features: int = 2000, ratio: float = 0.8, reproj_px: float = 2.0,
                 min_inliers: int = 30):
        self.depth_fn = depth_fn          # (left, right) -> depth in metres, 0 = invalid
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio, self.reproj_px, self.min_inliers = ratio, reproj_px, min_inliers

    def _features(self, rgb, tool_mask):
        keep = None if tool_mask is None else (~tool_mask).astype(np.uint8) * 255
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)  # endoscopic images are low-contrast and specular
        return self.orb.detectAndCompute(gray, keep)

    def relative(self, prev, cur, K, depth_prev) -> tuple[np.ndarray | None, int]:
        """T_prev_cur (pose of cur in prev's camera frame) and inlier count; None if tracking failed."""
        kp0, des0 = prev
        kp1, des1 = cur
        if des0 is None or des1 is None:
            return None, 0
        good = [m for m, n in (p for p in self.matcher.knnMatch(des0, des1, k=2) if len(p) == 2)
                if m.distance < self.ratio * n.distance]
        uv0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 2)
        uv1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)
        z = depth_prev[np.round(uv0[:, 1]).astype(int).clip(0, depth_prev.shape[0] - 1),
                       np.round(uv0[:, 0]).astype(int).clip(0, depth_prev.shape[1] - 1)]
        ok = z > 0
        if ok.sum() < self.min_inliers:
            return None, int(ok.sum())
        u, v, zz = uv0[ok, 0].astype(np.float64), uv0[ok, 1].astype(np.float64), z[ok].astype(np.float64)
        pts = np.stack([(u - K[0, 2]) * zz / K[0, 0], (v - K[1, 2]) * zz / K[1, 1], zz], 1)
        found, rvec, tvec, inl = cv2.solvePnPRansac(pts, uv1[ok], K, None, reprojectionError=self.reproj_px,
                                                    iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        if not found or inl is None or len(inl) < self.min_inliers:
            return None, 0 if inl is None else len(inl)
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(pts[inl], uv1[ok][inl], K, None, rvec, tvec)
        T_cur_prev = np.eye(4)
        T_cur_prev[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cur_prev[:3, 3] = tvec[:, 0]
        return np.linalg.inv(T_cur_prev), len(inl)

    def __call__(self, frames, K) -> tuple[np.ndarray, np.ndarray]:
        """Frames with .rgb/.right/.tool_mask -> (camera-to-world (N,4,4) starting at identity, inliers (N,)).

        A failed step reuses the previous relative motion (constant-velocity fallback) and records 0 inliers.
        """
        T = np.tile(np.eye(4), (len(frames), 1, 1))
        inliers = np.zeros(len(frames), int)
        last = np.eye(4)
        prev = self._features(frames[0].rgb, frames[0].tool_mask)
        depth_prev = self.depth_fn(frames[0].rgb, frames[0].right)
        for i in range(1, len(frames)):
            cur = self._features(frames[i].rgb, frames[i].tool_mask)
            rel, inliers[i] = self.relative(prev, cur, K, depth_prev)
            if rel is None:
                rel, inliers[i] = last, 0
            T[i] = T[i - 1] @ rel
            last = rel
            prev = cur
            depth_prev = self.depth_fn(frames[i].rgb, frames[i].right)
        return T, inliers
