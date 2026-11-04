import numpy as np
from scipy.spatial.transform import Rotation

from sr3d.eval.pose_metrics import ate_rmse, evaluate, rpe, umeyama


def trajectory(n=50, seed=0):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, 3] = np.cumsum(rng.normal(scale=0.01, size=(n, 3)), axis=0)
    T[:, :3, :3] = Rotation.from_rotvec(np.cumsum(rng.normal(scale=0.02, size=(n, 3)), axis=0)).as_matrix()
    return T


def sim3(T, s, R, t):
    out = T.copy()
    out[:, :3, :3] = R @ T[:, :3, :3]
    out[:, :3, 3] = s * T[:, :3, 3] @ R.T + t
    return out


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(100, 3))
    R = Rotation.random(random_state=2).as_matrix()
    s, R2, t = umeyama(src, 2.5 * src @ R.T + [1, 2, 3])
    np.testing.assert_allclose([s], [2.5])
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t, [1, 2, 3], atol=1e-9)


def test_metrics_invariant_to_similarity_transform():
    gt = trajectory()
    est = sim3(gt, 0.3, Rotation.random(random_state=5).as_matrix(), np.array([5.0, -1, 2]))
    m = evaluate(est, gt)
    assert m["ate_rmse_mm"] < 1e-6 and m["rpe_trans_mm"] < 1e-6 and m["rpe_rot_deg"] < 1e-6
    np.testing.assert_allclose(m["scale"], 1 / 0.3)


def test_metrics_detect_noise():
    gt = trajectory()
    est = gt.copy()
    est[:, :3, 3] += np.random.default_rng(3).normal(scale=0.001, size=(len(gt), 3))
    assert 0.5 < ate_rmse(est, gt, with_scale=False) * 1000 < 3.0
    assert rpe(est, gt)[0] > 0


def test_stereo_vo_recovers_metric_translation():
    import cv2
    from types import SimpleNamespace
    from sr3d.pose.stereo_vo import StereoVO

    # Textured surface whose depth varies by image row (5-12 cm); camera moves +x by 2 mm per frame.
    # Depth must vary: on a constant-depth plane, x-translation and yaw produce near-identical image
    # motion, so PnP trades ~0.1 deg of yaw for ~7% extra translation. An x-move keeps each point on
    # its row at the same depth, so frames render exactly by remapping texture columns.
    rng = np.random.default_rng(0)
    h, w, fx, tx = 240, 320, 400.0, 0.002
    z = np.linspace(0.05, 0.12, h, dtype=np.float32)[:, None]
    tex = cv2.GaussianBlur((rng.random((h, 2 * w)) * 255).astype(np.uint8), (3, 3), 0)
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
    rows = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, 1)

    def render(k):
        X = (np.arange(w) - K[0, 2]) * z / fx + k * tx      # world x seen by each pixel
        cols = (X * fx / 0.08 + w).astype(np.float32)         # texture pinned to the surface
        return cv2.cvtColor(cv2.remap(tex, cols, rows, cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB)

    frames = [SimpleNamespace(rgb=render(k), right=None, tool_mask=None) for k in range(3)]
    depth = np.broadcast_to(z, (h, w))
    T, inliers = StereoVO(lambda l, r: depth)(frames, K)
    assert (inliers[1:] > 30).all()
    # tolerances reflect ORB keypoint localisation noise (~0.25 px), not a scale bias
    np.testing.assert_allclose(T[1:, 0, 3], [tx, 2 * tx], atol=1e-4)
    np.testing.assert_allclose(T[1:, 1:3, 3], 0, atol=1.5e-4)
