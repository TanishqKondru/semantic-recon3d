import numpy as np
import pytest

from sr3d.ros.proximity import TissueModel, find_tip, measure, min_distance

K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
W, H = 64, 48
I4 = np.eye(4)


def tissue_plane(z=0.1, step=0.5e-3, size=0.1):
    g = np.arange(-size / 2, size / 2, step)
    x, y = np.meshgrid(g, g)
    return TissueModel(np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], 1))


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


def test_distance_to_plane_and_tip_depth():
    mask = bar_from_left()
    depth = np.where(mask, 0.095, 0.1).astype(np.float32)
    tools = measure(depth, mask, K, I4, tissue_plane(), erode_px=1, min_pixels=10)
    assert len(tools) == 1
    t = tools[0]
    assert t.distance == pytest.approx(5e-3, abs=3e-4)
    assert t.tip[2] == pytest.approx(0.095)
    assert min_distance(tools) == t.distance


def test_single_flying_pixel_does_not_set_the_minimum():
    mask = bar_from_left()
    depth = np.where(mask, 0.095, 0.1).astype(np.float32)
    depth[25, 20] = 0.0999                       # one bad stereo pixel almost on the tissue
    t = measure(depth, mask, K, I4, tissue_plane(), erode_px=1, min_pixels=10, median_px=5)[0]
    assert t.distance == pytest.approx(5e-3, abs=3e-4)
    raw = measure(depth, mask, K, I4, tissue_plane(), erode_px=1, min_pixels=10, median_px=1)[0]
    assert raw.distance < 1e-3


def test_depth_holes_are_eroded_like_mask_edges():
    mask = bar_from_left()
    depth = np.where(mask, 0.095, 0.1).astype(np.float32)
    depth[24:26, 15:17] = 0                      # hole in the stereo depth
    depth[23, 15:17] = 0.0999                    # bleed at the hole's rim
    t = measure(depth, mask, K, I4, tissue_plane(), erode_px=1, min_pixels=10, median_px=1)[0]
    assert t.distance == pytest.approx(5e-3, abs=3e-4)


def test_small_components_and_no_map():
    mask = np.zeros((H, W), bool)
    mask[5:7, 5:7] = True
    assert measure(np.full((H, W), 0.1, np.float32), mask, K, I4, None) == []
    tools = measure(np.full((H, W), 0.1, np.float32), bar_from_left(), K, I4, None, min_pixels=10)
    assert np.isnan(tools[0].distance) and np.isnan(min_distance(tools))


@pytest.mark.parametrize("gap", [0.5e-3, 5e-3, 40e-3])
def test_bounded_search_is_exact(gap):
    rng = np.random.default_rng(0)
    mask = bar_from_left()
    depth = np.where(mask, 0.1 - gap + rng.normal(0, 2e-4, (H, W)), 0.1).astype(np.float32)
    for px in (1, 5):
        t = measure(depth, mask, K, I4, tissue_plane(size=0.2), erode_px=1, min_pixels=10, median_px=px)[0]
        brute = np.abs(t.points[:, 2] - 0.1)    # dense plane: nearest-point distance ~ vertical distance
        expected = brute.min() if px == 1 else gap
        assert t.distance == pytest.approx(expected, abs=3e-4 if px == 1 else 1e-3)
