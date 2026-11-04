import numpy as np
import open3d as o3d
import pytest

from sr3d.eval.fusion_metrics import MeshRenderer, heldout_scores
from sr3d.fusion.labels import label_iou, vote
from sr3d.fusion.tsdf import tissue_depth

K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
W, H = 64, 48


def plane_mesh(z=0.1, size=0.2, color=(0.5, 0.25, 0.75)):
    s = size / 2
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]]),
        o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
    m.vertex_colors = o3d.utility.Vector3dVector([color] * 4)
    return m


def test_tissue_depth_removes_dilated_instrument():
    depth = np.full((H, W), 0.1, np.float32)
    tool = np.zeros((H, W), bool)
    tool[20:24, 30:34] = True
    out = tissue_depth(depth, tool, dilate_px=2)
    assert (out[tool] == 0).all() and out[20, 28] == 0 and out[20, 26] == pytest.approx(0.1)
    assert tissue_depth(depth, None) is depth


def test_renderer_depth_and_colour_on_a_plane():
    rd, rgb = MeshRenderer(plane_mesh()).render(K, np.eye(4), W, H)
    assert np.allclose(rd, 0.1, atol=1e-5)  # depth along the optical axis, not ray length
    assert np.allclose(rgb, [127.5, 63.75, 191.25], atol=0.5)
    s = heldout_scores(rd, rgb, np.full((H, W), 0.101, np.float32), rgb.astype(np.uint8), np.ones((H, W), bool))
    assert s["depth_mm"] == pytest.approx(1.0, abs=1e-2) and s["coverage"] == 1.0 and s["psnr"] > 40


def test_vote_respects_visibility_and_averages():
    pts = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])  # second point hidden behind the first
    depth = np.full((H, W), 0.1, np.float32)
    frames = [(np.full((H, W), p), depth, np.eye(4)) for p in (1.0, 0.0, 1.0)]
    v = vote(pts, frames, K)
    assert v.prob[0] == pytest.approx(2 / 3) and v.last[0] == 1.0 and v.seen[0] == 3
    assert np.isnan(v.prob[1]) and v.seen[1] == 0


def test_label_iou_ignores_unseen():
    pred = np.array([0.9, 0.1, np.nan, 0.8])
    ref = np.array([0.9, 0.9, 0.9, 0.1])
    assert label_iou(pred, ref) == pytest.approx(1 / 3)
