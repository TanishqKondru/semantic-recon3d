"""3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) of the static tissue with gsplat (Ye et al., JMLR 2025).

Gaussians are initialised from the fused TSDF surface and optimised on the posed fusion frames. Instrument
pixels are masked out of every loss term, since instruments move and are not part of the static map.
Stereo depth supervises the rendered expected depth so geometry stays metric between views, which a
photometric loss alone does not ensure with the small baseline of an endoscope.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

SH_C0 = 0.28209479177387814


def ssim_map(a: torch.Tensor, b: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-pixel SSIM of (1, 3, H, W) images in [0, 1], averaged over channels -> (H, W)."""
    x = torch.arange(window, device=a.device, dtype=a.dtype) - window // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None]
    k = g.expand(3, 1, window, window)
    blur = lambda t: F.conv2d(t, k, padding=window // 2, groups=3)
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.mean(1)[0]


class GaussianMap:
    def __init__(self, points: np.ndarray, colors: np.ndarray, scene_scale: float, sh_degree: int = 3,
                 device: str = "cuda"):
        """points (N, 3) metres, colors (N, 3) in [0, 1]; scene_scale = radius of the camera path (m)."""
        self.device, self.sh_degree, self.scene_scale = device, sh_degree, scene_scale
        pts = torch.tensor(points, dtype=torch.float32, device=device)
        # initial size: mean distance to the 3 nearest neighbours (chunked to bound memory)
        nn = []
        for chunk in torch.split(pts, 4096):
            nn.append(torch.cdist(chunk, pts).topk(4, largest=False).values[:, 1:].mean(1))
        dist = torch.cat(nn).clamp_min(1e-5)
        n = len(pts)
        sh = torch.zeros(n, (sh_degree + 1) ** 2, 3, device=device)
        sh[:, 0] = (torch.tensor(colors, dtype=torch.float32, device=device) - 0.5) / SH_C0
        self.params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(pts),
            "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=device), dim=-1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=device))),
            "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
            "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
        })

    def render(self, K: np.ndarray, T_wc: np.ndarray, width: int, height: int, sh_degree: int | None = None):
        """-> rgb (H, W, 3) in [0, 1], expected depth (H, W) metres, alpha (H, W), gsplat info."""
        p = self.params
        viewmat = torch.tensor(np.linalg.inv(T_wc), dtype=torch.float32, device=self.device)[None]
        Ks = torch.tensor(K, dtype=torch.float32, device=self.device)[None]
        out, alpha, info = rasterization(
            p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
            torch.cat([p["sh0"], p["shN"]], 1), viewmat, Ks, width, height,
            sh_degree=self.sh_degree if sh_degree is None else sh_degree,
            near_plane=1e-3, far_plane=1.0, render_mode="RGB+ED", packed=False)
        return out[0, ..., :3].clamp(0, 1), out[0, ..., 3], alpha[0, ..., 0], info

    def fit(self, frames, K: np.ndarray, width: int, height: int, iters: int = 7000, depth_weight: float = 0.05,
            ssim_weight: float = 0.2, log_every: int = 1000, seed: int = 0) -> list[dict]:
        """frames: list of (rgb uint8 (H, W, 3), depth metres (H, W), tissue mask (H, W) bool, T_wc)."""
        s = self.scene_scale
        lrs = {"means": 1.6e-4 * s, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        opt = {k: torch.optim.Adam([{"params": self.params[k], "lr": lr, "name": k}], eps=1e-15)
               for k, lr in lrs.items()}
        sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1 / iters))
        strategy = DefaultStrategy(refine_stop_iter=iters // 2, reset_every=3000, verbose=False)
        strategy.check_sanity(self.params, opt)
        state = strategy.initialize_state(scene_scale=s)
        data = [(torch.tensor(rgb, device=self.device, dtype=torch.float32) / 255,
                 torch.tensor(depth, device=self.device, dtype=torch.float32),
                 torch.tensor(tissue, device=self.device), T) for rgb, depth, tissue, T in frames]
        rng = np.random.default_rng(seed)
        log = []
        for step in range(iters):
            rgb, depth, tissue, T = data[rng.integers(len(data))]
            deg = min(step // 1000, self.sh_degree)
            img, d, alpha, info = self.render(K, T, width, height, sh_degree=deg)
            strategy.step_pre_backward(self.params, opt, state, step, info)
            m = tissue.float()[..., None]
            l1 = (torch.abs(img - rgb) * m).sum() / (3 * m.sum())
            ssim = (ssim_map(img.permute(2, 0, 1)[None], rgb.permute(2, 0, 1)[None]) * m[..., 0]).sum() / m.sum()
            dm = tissue & (depth > 0)
            ld = 1000 * torch.abs(d - depth)[dm].mean()  # mm
            loss = (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim) + depth_weight * ld
            loss.backward()
            strategy.step_post_backward(self.params, opt, state, step, info, packed=False)
            for o in opt.values():
                o.step()
                o.zero_grad(set_to_none=True)
            sched.step()
            if (step + 1) % log_every == 0:
                log.append({"step": step + 1, "l1": float(l1), "ssim": float(ssim), "depth_mm": float(ld),
                            "gaussians": len(self.params["means"])})
                print(f"  step {step + 1}: l1 {float(l1):.4f}  ssim {float(ssim):.3f}  depth {float(ld):.2f} mm  "
                      f"{len(self.params['means'])} gaussians", flush=True)
        return log

    @torch.no_grad()
    def render_numpy(self, K, T_wc, width, height, min_alpha: float = 0.5):
        """Rendered depth (0 where alpha < min_alpha) and RGB in [0, 255], as numpy, for scoring."""
        img, d, alpha, _ = self.render(K, T_wc, width, height)  # "ED" depth is already normalised by alpha
        d = torch.where(alpha > min_alpha, d, torch.zeros_like(d))
        return d.cpu().numpy(), img.cpu().numpy() * 255

    def centers(self) -> np.ndarray:
        return self.params["means"].detach().cpu().numpy()


def camera_radius(poses: np.ndarray) -> float:
    c = poses[:, :3, 3]
    return float(np.linalg.norm(c - c.mean(0), axis=1).max() * 1.1) or math.inf
