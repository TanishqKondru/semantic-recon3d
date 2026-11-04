import numpy as np

from sr3d.depth.sgbm import SGBMStereo
from sr3d.eval.depth_metrics import depth_metrics


def synthetic_pair(disp: int = 12, h: int = 120, w: int = 200):
    rng = np.random.default_rng(0)
    tex = (rng.random((h, w + disp)) * 255).astype(np.uint8)
    tex = np.repeat(tex[..., None], 3, axis=2)
    left = tex[:, :w]
    right = tex[:, disp:disp + w]  # left column u shows tex[u]; right shows it at u - disp
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def test_sgbm_recovers_constant_disparity():
    left, right = synthetic_pair(disp=12)
    d = SGBMStereo(num_disparities=32, block_size=5).disparity(left, right)
    core = d[10:-10, 40:-10]
    valid = core > 0
    assert valid.mean() > 0.8
    assert np.median(np.abs(core[valid] - 12)) < 0.5


def test_depth_metrics_perfect_and_scaled():
    gt = np.full((10, 10), 0.1)
    m = depth_metrics(gt.copy(), gt)
    assert m["abs_rel"] == 0 and m["delta1"] == 1.0
    m = depth_metrics(gt * 3, gt, align_scale=True)
    assert abs(m["abs_rel"]) < 1e-12 and abs(m["scale"] - 1 / 3) < 1e-12


def test_depth_metrics_mm_units():
    gt = np.full((4, 4), 0.100)
    m = depth_metrics(gt + 0.002, gt)
    assert abs(m["mae_mm"] - 2.0) < 1e-9
