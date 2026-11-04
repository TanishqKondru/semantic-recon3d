from sr3d.segmentation.data import select_frames


def test_select_frames_splits_tool_and_empty_and_drops_trocar():
    # frames 0-9 tool-free, 10-29 with a tool, 30-34 inside the trocar (all metal)
    fractions = [(f, 0.0) for f in range(10)] + [(f, 0.1) for f in range(10, 30)] + [(f, 1.0) for f in range(30, 35)]
    frames = select_frames(fractions, n_tool=5, n_empty=2)
    assert frames == [0, 9, 10, 15, 20, 24, 29]


def test_select_frames_keeps_everything_under_budget():
    fractions = [(0, 0.0), (1, 0.2), (2, 0.96)]
    assert select_frames(fractions, n_tool=10, n_empty=10) == [0, 1]


def test_seg_meter_counts_pixels_and_false_alarms():
    import numpy as np
    from sr3d.eval.seg_metrics import SegMeter

    gt = np.zeros((10, 10), bool)
    gt[:5] = True  # 50 tool pixels
    pred = np.zeros((10, 10), bool)
    pred[:4] = True  # 40 correct, 10 missed
    pred[9, :5] = True  # 5 false positives
    m = SegMeter()
    m.update(pred, gt)
    m.update(np.ones((10, 10), bool), np.zeros((10, 10), bool))  # tool-free frame, all predicted tool
    m.update(np.zeros((10, 10), bool), np.zeros((10, 10), bool))  # tool-free frame, clean
    s = m.summary()
    assert s["n_tool_frames"] == 1 and s["n_empty_frames"] == 2
    assert s["empty_false_alarm_rate"] == 0.5
    assert abs(s["mean_frame_iou"] - 40 / 55) < 1e-9
    assert abs(s["iou"] - 40 / 155) < 1e-9  # dataset-level counts include the false-alarm frame


def test_seg_dataset_augment_keeps_image_and_mask_aligned(tmp_path):
    import cv2
    import numpy as np
    from sr3d.segmentation.data import SegDataset

    img = np.zeros((96, 128, 3), np.uint8)
    img[20:60, 30:90] = 255  # white box
    mask = np.zeros((96, 128), np.uint8)
    mask[20:60, 30:90] = 255  # same box as the instrument
    cv2.imwrite(str(tmp_path / "i.png"), img)
    cv2.imwrite(str(tmp_path / "m.png"), mask)
    ds = SegDataset([(tmp_path / "i.png", tmp_path / "m.png")], crop=64, seed=1)
    for _ in range(20):
        x, y = ds[0]
        assert x.shape == (3, 64, 64) and y.shape == (64, 64)
        bright = x.mean(0) > 0.5
        assert (bright == y.astype(bool)).mean() > 0.95


def test_simulated_fluorescence_is_detected_and_spares_instruments():
    import numpy as np
    from sr3d.segmentation.data import is_fluorescence, simulate_fluorescence

    rng = np.random.default_rng(0)
    img = np.zeros((128, 160, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 180, 60, 50  # red tissue
    img[40:80, 60:100] = 150  # grey instrument
    mask = np.zeros((128, 160), np.uint8)
    mask[40:80, 60:100] = 1
    assert not is_fluorescence(img)
    for seed in range(5):
        out = simulate_fluorescence(img, mask, np.random.default_rng(seed)).astype(int)
        assert is_fluorescence(out.astype(np.uint8))
        tool = out[45:75, 65:95]
        assert np.abs(tool[..., 1] - tool[..., 0]).max() <= 2  # instrument stays grey, no glow


def test_tracking_helpers():
    import numpy as np
    from sr3d.segmentation.tracking import flicker, keyframes

    assert keyframes(10, 0) == [0]
    assert keyframes(10, 4) == [0, 4, 8]
    a = np.zeros((4, 4), bool)
    b = a.copy()
    b[:2] = True
    c = b.copy()
    c[:, :2] = True  # union 12, intersection 8 with b
    assert flicker([a, a]) == 0.0
    assert flicker([b, b, c]) == (0 + (1 - 8 / 12)) / 2


def test_confident_keyframes_skips_unsure_windows():
    from sr3d.segmentation.tracking import confident_keyframes

    u = [0.5, 0.1, 0.3, 0.4, 0.9, 0.6, 0.05, 0.2]
    # windows of 2: [0,1] -> 1 (0.1); [2,3] -> 2 (0.3 > 0.25, skipped); [4,5] -> 5 (0.6, skipped); [6,7] -> 6
    assert confident_keyframes(u, 2) == [1, 6]
    # the first window is always prompted, even when unsure
    assert confident_keyframes([0.9, 0.8, 0.1], 2) == [1, 2]
