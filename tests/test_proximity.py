import numpy as np
import pytest

from sr3d.ros.proximity import TissueModel, find_tip, measure, min_distance

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False
BACKENDS = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA"))]


def model_cls(backend):
    if backend == "gpu":
        from sr3d.ros.gpu_nn import GPUTissueModel
        return GPUTissueModel
    return TissueModel

K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
W, H = 64, 48
I4 = np.eye(4)


def tissue_plane(z=0.1, step=0.5e-3, size=0.1, backend="cpu"):
    g = np.arange(-size / 2, size / 2, step)
    x, y = np.meshgrid(g, g)
    return model_cls(backend)(np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], 1))


def bar_from_left(length=40):
    m = np.zeros((H, W), bool)
    m[20:30, :length] = True
    return m


def test_tip_is_far_end_of_bar_entering_from_border():
    assert find_tip(bar_from_left()) == (39, 20)


def test_tip_out_of_view_when_tool_spans_the_image():
    assert find_tip(bar_from_left(W)) is None


def test_tip_of_floating_component_points_away_from_nearest_edge():
    m = np.zeros((H, W), bool)
    m[20:24, 3:30] = True     # centroid is nearest the left edge
    assert find_tip(m)[0] == 29


@pytest.mark.parametrize("backend", BACKENDS)
def test_distance_to_plane_and_tip_depth(backend):
    mask = bar_from_left()
    depth = np.where(mask, 0.095, 0.1).astype(np.float32)
    tools = measure(depth, mask, K, I4, tissue_plane(backend=backend), erode_px=1, min_pixels=10)
    assert len(tools) == 1
    t = tools[0]
    assert t.distance == pytest.approx(5e-3, abs=3e-4)
    assert t.tip[2] == pytest.approx(0.095)
    assert min_distance(tools) == t.distance


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_flying_pixel_does_not_set_the_minimum(backend):
    mask = bar_from_left()
    depth = np.where(mask, 0.095, 0.1).astype(np.float32)
    depth[25, 20] = 0.0999                       # one bad stereo pixel almost on the tissue
    t = measure(depth, mask, K, I4, tissue_plane(backend=backend), erode_px=1, min_pixels=10, median_px=5)[0]
    assert t.distance == pytest.approx(5e-3, abs=3e-4)
    raw = measure(depth, mask, K, I4, tissue_plane(backend=backend), erode_px=1, min_pixels=10, median_px=1)[0]
    assert raw.distance < 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_depth_holes_are_eroded_like_mask_edges(backend):
    mask = bar_from_left()
    depth = np.where(mask, 0.095, 0.1).astype(np.float32)
    depth[24:26, 15:17] = 0                      # hole in the stereo depth
    depth[23, 15:17] = 0.0999                    # bleed at the hole's rim
    t = measure(depth, mask, K, I4, tissue_plane(backend=backend), erode_px=1, min_pixels=10, median_px=1)[0]
    assert t.distance == pytest.approx(5e-3, abs=3e-4)


def test_small_components_and_no_map():
    mask = np.zeros((H, W), bool)
    mask[5:7, 5:7] = True
    assert measure(np.full((H, W), 0.1, np.float32), mask, K, I4, None) == []
    tools = measure(np.full((H, W), 0.1, np.float32), bar_from_left(), K, I4, None, min_pixels=10)
    assert np.isnan(tools[0].distance) and np.isnan(min_distance(tools))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("gap", [0.5e-3, 5e-3, 40e-3])
def test_bounded_search_is_exact(gap, backend):
    rng = np.random.default_rng(0)
    mask = bar_from_left()
    depth = np.where(mask, 0.1 - gap + rng.normal(0, 2e-4, (H, W)), 0.1).astype(np.float32)
    for px in (1, 5):
        t = measure(depth, mask, K, I4, tissue_plane(size=0.2, backend=backend), erode_px=1, min_pixels=10, median_px=px)[0]
        brute = np.abs(t.points[:, 2] - 0.1)    # dense plane: nearest-point distance ~ vertical distance
        expected = brute.min() if px == 1 else gap
        assert t.distance == pytest.approx(expected, abs=3e-4 if px == 1 else 1e-3)


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
@pytest.mark.parametrize("upper", [np.inf, 3e-3, 1e-3, 1e-5])
def test_gpu_nearest_matches_kd_tree(upper):
    """Random scene-scale clouds (two sets, different origins): same distances and neighbours as the k-d tree."""
    from sr3d.ros.gpu_nn import GPUTissueModel
    rng = np.random.default_rng(1)
    a = rng.uniform(-0.03, 0.03, (20_000, 3)) + [0, 0, 0.1]
    b = rng.uniform(-0.03, 0.03, (7_000, 3)) + [0.01, 0, 0.09]
    q = rng.uniform(-0.035, 0.035, (3_001, 3)) + [0, 0, 0.1]
    d_cpu, p_cpu = TissueModel.merge(TissueModel(a), TissueModel(b)).query(q, upper)
    d_gpu, p_gpu = GPUTissueModel.merge(GPUTissueModel(a), GPUTissueModel(b)).query(q, upper)
    assert np.array_equal(np.isinf(d_cpu), np.isinf(d_gpu))
    f = np.isfinite(d_cpu)
    if not f.any():
        return
    assert np.abs(d_cpu[f] - d_gpu[f]).max() < 1e-7          # 0.1 micrometre
    assert np.abs(p_cpu[f] - p_gpu[f]).max() < 1e-6          # same neighbour (float32 storage of the point)


def half_cylinder(gap, r=4e-3, z_plane=0.1, length=0.04, tilt=0.3, noise=0.0, seed=0):
    """Camera-facing half of a capsule hovering `gap` above the plane z = z_plane (camera at the origin)."""
    rng = np.random.default_rng(seed)
    a = np.array([np.cos(tilt), 0, np.sin(tilt)])             # shaft -> tip, descending towards the plane (+z)
    tip_c = np.array([0.0, 0.0, z_plane - r - gap])
    s = rng.uniform(0, length, 20000)
    phi = rng.uniform(0, 2 * np.pi, 20000)
    x = np.cross(a, [0, 1.0, 0]); x /= np.linalg.norm(x); y = np.cross(a, x)
    n = np.cos(phi)[:, None] * x + np.sin(phi)[:, None] * y
    p = tip_c - s[:, None] * a + r * n                       # shaft extends away from the tip
    vis = (n * -p).sum(1) > 0                                 # normal faces the camera
    p = p[vis] + rng.normal(0, noise, (vis.sum(), 3))
    return p, tip_c, a


@pytest.mark.parametrize("gap", [0.5e-3, 2e-3, 5e-3])
def test_capsule_fit_recovers_hidden_underside(gap):
    from sr3d.ros.toolmodel import fit_capsule
    pts, tip_c, a = half_cylinder(gap, noise=0.1e-3)
    plane = tissue_plane(z=0.1, size=0.12)
    visible = plane.query(pts)[0].min()
    assert visible > gap + 2e-3                               # the camera-facing half alone reads far too high
    cap = fit_capsule(pts, 4e-3, np.zeros(3), tip=tip_c)
    assert cap is not None and cap.rms < 0.3e-3
    assert abs(abs(cap.axis @ a) - 1) < 1e-3
    d = plane.query(cap.surface_samples(0.1e-3))[0].min()
    assert d == pytest.approx(gap, abs=0.3e-3)


def test_capsule_fit_rejects_blobs():
    from sr3d.ros.toolmodel import fit_capsule
    rng = np.random.default_rng(0)
    blob = rng.normal(0, 2e-3, (2000, 3)) + [0, 0, 0.09]      # not a cylinder, not elongated
    assert fit_capsule(blob, 4e-3, np.zeros(3)) is None


@pytest.mark.parametrize("gap", [0.5e-3, 3e-3])
def test_offset_capsule_is_cautious_and_close(gap):
    """Offset model on a true cylinder: never reads farther than the truth, and within 2 mm of it."""
    from sr3d.ros.toolmodel import offset_capsule
    pts, tip_c, _ = half_cylinder(gap, noise=0.1e-3)
    cap = offset_capsule(pts, 4e-3, np.zeros(3), tip=tip_c)
    d = tissue_plane(z=0.1, size=0.12).query(cap.surface_samples(0.1e-3))[0].min()
    assert gap - 2e-3 <= d <= gap + 0.1e-3
