import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sr3d.data.poses import read_tum, relative_to_first, write_tum
from sr3d.data.scene import Scene, SceneMeta, SceneWriter
from sr3d.geometry import backproject, disparity_to_depth, project

K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])


def random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.random(n, random_state=seed).as_matrix()
    T[:, :3, 3] = rng.normal(size=(n, 3))
    return T


def test_tum_roundtrip(tmp_path):
    T = random_poses(5)
    write_tum(tmp_path / "p.txt", np.arange(5.0), T)
    stamps, T2 = read_tum(tmp_path / "p.txt")
    np.testing.assert_allclose(stamps, np.arange(5.0))
    np.testing.assert_allclose(T2, T, atol=1e-7)


def test_relative_to_first():
    T = relative_to_first(random_poses(4))
    np.testing.assert_allclose(T[0], np.eye(4), atol=1e-12)


def test_backproject_project_inverse():
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    T_wc = random_poses(1, seed=3)[0]
    pts, vu = backproject(depth, K, T_wc)
    uv, z = project(pts, K, T_wc)
    np.testing.assert_allclose(uv, vu[:, ::-1], atol=1e-6)
    np.testing.assert_allclose(z, 2.0, atol=1e-6)


def test_backproject_respects_mask_and_invalid_depth():
    depth = np.ones((4, 4), dtype=np.float32)
    depth[0, 0] = 0
    mask = np.ones((4, 4), bool)
    mask[1, 1] = False
    pts, _ = backproject(depth, K, mask=mask)
    assert len(pts) == 14


def test_disparity_to_depth():
    d = disparity_to_depth(np.array([[10.0, 0.1]]), fx=500.0, baseline_m=0.005)
    np.testing.assert_allclose(d, [[0.25, 0.0]])


def test_scene_roundtrip(tmp_path):
    meta = SceneMeta(name="t", domain="medical", source="synthetic", width=64, height=48, K=K.tolist())
    T = random_poses(3)
    rng = np.random.default_rng(0)
    rgbs = rng.integers(0, 255, size=(3, 48, 64, 3), dtype=np.uint8)
    depth = rng.uniform(0.05, 0.2, size=(48, 64)).astype(np.float32)
    mask = rng.random((48, 64)) > 0.5
    with SceneWriter(tmp_path / "s", meta) as w:
        for i in range(3):
            w.add(rgbs[i], T_wc=T[i], depth=depth, tool_mask=mask)

    scene = Scene(tmp_path / "s")
    assert len(scene) == 3 and scene.has("depth") and not scene.has("right")
    f = scene[2]
    np.testing.assert_array_equal(f.rgb, rgbs[2])
    np.testing.assert_allclose(f.T_wc, T[2], atol=1e-7)
    np.testing.assert_allclose(f.depth, depth, atol=0.5 / meta.depth_scale)  # uint16 mm quantisation
    np.testing.assert_array_equal(f.tool_mask, mask)


def test_writer_rejects_wrong_size(tmp_path):
    meta = SceneMeta(name="t", domain="medical", source="x", width=64, height=48, K=K.tolist())
    w = SceneWriter(tmp_path / "s", meta)
    with pytest.raises(ValueError):
        w.add(np.zeros((10, 10, 3), np.uint8))


def test_writer_rejects_partial_poses(tmp_path):
    meta = SceneMeta(name="t", domain="medical", source="x", width=64, height=48, K=K.tolist())
    w = SceneWriter(tmp_path / "s", meta)
    w.add(np.zeros((48, 64, 3), np.uint8), T_wc=np.eye(4))
    with pytest.raises(ValueError):
        w.add(np.zeros((48, 64, 3), np.uint8))


def test_stereomis_rectifier_recovers_baseline():
    from sr3d.data.converters.stereomis import Rectifier

    Kc = np.array([[500.0, 0, 320], [0, 500.0, 256], [0, 0, 1]])
    calib = {"size": (640, 512), "K1": Kc, "K2": Kc, "d1": np.zeros(8), "d2": np.zeros(8),
             "R": np.eye(3), "T": np.array([-5.0, 0, 0])}  # right camera 5 mm to the right, in mm
    rect = Rectifier(calib, out_width=320)
    assert rect.size == (320, 256)
    np.testing.assert_allclose(rect.baseline_m, 0.005, rtol=1e-6)
    np.testing.assert_allclose(rect.K[0, 0], 250.0, rtol=1e-6)
    np.testing.assert_allclose(rect.pose_to_rectified(np.eye(4)), np.eye(4), atol=1e-9)
    left = np.random.default_rng(0).integers(0, 255, (512, 640, 3), dtype=np.uint8)
    l2, r2 = rect(left, left)
    assert l2.shape == (256, 320, 3) and r2.shape == (256, 320, 3)


def test_stereomis_mask_decoding_handles_both_encodings():
    from sr3d.data.converters.stereomis import decode_mask

    tool = np.zeros((4, 4), bool)
    tool[1:3, 1:3] = True
    rgb_0_255 = np.repeat(np.where(tool, 0, 255).astype(np.uint8)[..., None], 3, -1)  # most sequences
    gray_0_1 = np.where(tool, 0, 1).astype(np.uint8)                                     # P3
    np.testing.assert_array_equal(decode_mask(rgb_0_255), tool)
    np.testing.assert_array_equal(decode_mask(gray_0_1), tool)
