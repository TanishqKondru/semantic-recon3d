"""Smoke-test the GPU stack: PyTorch, gsplat, SAM 2, Depth Anything V2, Open3D."""
import importlib
import time

import torch


def check(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        print(f"[ OK ] {name:<20} {msg}  ({time.time() - t0:.1f}s)")
    except Exception as e:  # report and keep going so we see every failure at once
        print(f"[FAIL] {name:<20} {type(e).__name__}: {e}")


def torch_gpu():
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    return f"{torch.__version__} cuda={torch.version.cuda} {torch.cuda.get_device_name(0)}"


def gsplat_render():
    from gsplat import rasterization

    n = 10_000
    means = torch.randn(n, 3, device="cuda") * 0.5 + torch.tensor([0, 0, 3.0], device="cuda")
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
    scales = torch.full((n, 3), 0.02, device="cuda")
    opacities = torch.full((n,), 0.8, device="cuda")
    colors = torch.rand(n, 3, device="cuda")
    K = torch.tensor([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], device="cuda")[None]
    viewmat = torch.eye(4, device="cuda")[None]
    img, alpha, _ = rasterization(means, quats, scales, opacities, colors, viewmat, K, 320, 240)
    return f"rendered {tuple(img.shape)}, mean alpha={alpha.mean().item():.2f}"


def depth_anything():
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
    from PIL import Image
    import numpy as np

    out = pipe(Image.fromarray((np.random.rand(256, 320, 3) * 255).astype("uint8")))
    return f"depth map {out['predicted_depth'].shape}"


def sam2():
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import numpy as np

    pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small", device="cuda")
    pred.set_image((np.random.rand(256, 320, 3) * 255).astype("uint8"))
    masks, scores, _ = pred.predict(point_coords=np.array([[160, 128]]), point_labels=np.array([1]))
    return f"{masks.shape[0]} masks, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB"


def open3d():
    o3d = importlib.import_module("open3d")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(torch.rand(1000, 3).numpy()))
    return f"{o3d.__version__}, {len(pcd.points)} pts"


if __name__ == "__main__":
    for name, fn in [("torch+cuda", torch_gpu), ("gsplat", gsplat_render),
                     ("depth-anything-v2", depth_anything), ("sam2.1", sam2), ("open3d", open3d)]:
        check(name, fn)
