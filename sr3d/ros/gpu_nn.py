"""Exact nearest-neighbour search on the GPU for the proximity query (Triton kernel, no tree to build).

The CPU path builds a k-d tree for the live frame's tissue every frame (~80k points) and for the fused map on
every refresh, then queries it. On the GPU a brute-force search is cheaper than building a tree: one fused
kernel streams the tissue points through on-chip memory and keeps a running minimum per instrument point,
so the N x M distance matrix is never written out. Squared distances are computed from coordinate
differences (not |a|^2 + |b|^2 - 2ab), so there is no cancellation at sub-millimetre gaps and the result
matches the k-d tree to float32 rounding (~1e-8 m at 0.1 m depth).

With a finite bound, tissue outside the instruments' bounding box grown by the bound is skipped first; any
pair it removes is farther apart than the bound, so the bounded result is unchanged.

`GPUTissueModel` has the interface of `sr3d.ros.proximity.TissueModel` (numpy in, numpy out).
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _nn_kernel(q_ptr, t_ptr, d_ptr, i_ptr, n, m, BLOCK_Q: tl.constexpr, BLOCK_T: tl.constexpr):
    # q: (3, n) and t: (3, m) float32, structure of arrays; out: squared distance and index per query
    pid = tl.program_id(0)
    qo = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qm = qo < n
    qx = tl.load(q_ptr + qo, mask=qm, other=0.0)
    qy = tl.load(q_ptr + n + qo, mask=qm, other=0.0)
    qz = tl.load(q_ptr + 2 * n + qo, mask=qm, other=0.0)
    best = tl.full((BLOCK_Q,), float("inf"), tl.float32)
    best_i = tl.zeros((BLOCK_Q,), tl.int32)
    for start in range(0, m, BLOCK_T):
        to = start + tl.arange(0, BLOCK_T)
        tm = to < m
        tx = tl.load(t_ptr + to, mask=tm, other=float("inf"))
        ty = tl.load(t_ptr + m + to, mask=tm, other=float("inf"))
        tz = tl.load(t_ptr + 2 * m + to, mask=tm, other=float("inf"))
        dx = qx[:, None] - tx[None, :]
        dy = qy[:, None] - ty[None, :]
        dz = qz[:, None] - tz[None, :]
        d2 = dx * dx + dy * dy + dz * dz
        blk = tl.min(d2, axis=1)
        blk_i = tl.argmin(d2, axis=1).to(tl.int32) + start
        closer = blk < best
        best = tl.where(closer, blk, best)
        best_i = tl.where(closer, blk_i, best_i)
    tl.store(d_ptr + qo, best, mask=qm)
    tl.store(i_ptr + qo, best_i, mask=qm)


def nearest(q: torch.Tensor, t: torch.Tensor, block_q: int = 64, block_t: int = 128
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """q (N, 3), t (M, 3) float32 CUDA -> (distance (N,), index into t (N,)). M must be > 0."""
    n, m = len(q), len(t)
    d2 = torch.empty(n, device=q.device, dtype=torch.float32)
    idx = torch.empty(n, device=q.device, dtype=torch.int32)
    if n:
        _nn_kernel[(triton.cdiv(n, block_q),)](q.T.contiguous(), t.T.contiguous(), d2, idx, n, m,
                                              BLOCK_Q=block_q, BLOCK_T=block_t)
    return d2.sqrt(), idx.long()


def warmup(device: str = "cuda") -> None:
    """Compile the kernel (Triton JIT, ~1 s on first use) before the first frame that needs it."""
    q = torch.zeros((1, 3), device=device)
    nearest(q, q)
    torch.cuda.synchronize()


class GPUTissueModel:
    """Drop-in for `TissueModel` that keeps the tissue points on the GPU and searches them by brute force.

    Coordinates are stored relative to a fixed origin (the first point set's mean) so float32 keeps
    ~10 nm resolution over a 0.1 m scene."""

    def __init__(self, *point_sets: np.ndarray, device: str = "cuda"):
        self.device = device
        sets = [np.asarray(p, np.float64) for p in point_sets if len(p)]
        self.origin = sets[0].mean(0) if sets else np.zeros(3)
        self.sets = [self._upload(p) for p in sets]

    def _upload(self, p: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((p - self.origin).astype(np.float32)).to(self.device)

    @classmethod
    def merge(cls, *models: GPUTissueModel | None) -> GPUTissueModel:
        """Combine models (no copy of the point sets, apart from re-centring on the first model's origin)."""
        models = [m for m in models if m is not None]
        out = cls(device=models[0].device if models else "cuda")
        if models:
            out.origin = models[0].origin
            for m in models:
                shift = torch.from_numpy((m.origin - out.origin).astype(np.float32)).to(out.device)
                out.sets += [s + shift if shift.any() else s for s in m.sets]
        return out

    def __len__(self) -> int:
        return sum(len(p) for p in self.sets)

    def query(self, pts: np.ndarray, upper: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
        """-> (distance (N,), nearest tissue point (N, 3)); distances beyond `upper` come back as inf."""
        n = len(pts)
        best_d = torch.full((n,), float("inf"), device=self.device)
        best_p = torch.zeros((n, 3), device=self.device)
        if n and self.sets:
            q = self._upload(np.asarray(pts, np.float64))
            lo, hi = q.min(0).values, q.max(0).values
            for s in self.sets:
                if np.isfinite(upper):
                    keep = ((s >= lo - upper) & (s <= hi + upper)).all(1)
                    s = s[keep]
                if len(s) == 0:
                    continue
                d, j = nearest(q, s)
                closer = d < best_d
                best_d = torch.where(closer, d, best_d)
                best_p = torch.where(closer[:, None], s[j], best_p)
        if np.isfinite(upper):
            best_d = torch.where(best_d <= upper, best_d, torch.full_like(best_d, float("inf")))
        d = best_d.cpu().numpy().astype(np.float64)
        p = best_p.cpu().numpy().astype(np.float64) + self.origin
        return d, p
