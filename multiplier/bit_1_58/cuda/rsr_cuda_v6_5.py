"""
RSR ternary CUDA v6.5 — v6.3 with grid-stride loop and tuned grid size.

Reduces kernel waves by launching fewer CUDA blocks that process multiple
RSR blocks each.  Grid size is auto-tuned on first use (tries powers-of-2
fractions of num_blocks and picks the fastest).
"""

import numpy as np
import torch

from multiplier.base import Multiplier
from ._jit_build import load_kernel
from .rsr_cuda_v6_0 import pack_weights_int2

_module = load_kernel("rsr_ternary_cuda_v6_5", "rsr_ternary_v6_5.cu")


def _auto_tune_grid(packed_w, v_i8, inv_scale, out, n, k, num_blocks, k_dim, device):
    """Try several grid sizes and return the fastest."""
    # Candidate grid sizes: powers-of-2 divisors and the full num_blocks
    candidates = sorted(set([num_blocks, num_blocks // 2, num_blocks // 4,
                              num_blocks // 8, num_blocks // 16, 512, 1024, 2048]))
    candidates = [g for g in candidates if 64 <= g <= num_blocks]
    if not candidates:
        return num_blocks

    v = torch.randn(n, dtype=torch.float32, device=device)

    best_time = float('inf')
    best_grid = num_blocks
    for grid in candidates:
        # Warm up
        for _ in range(3):
            _module.rsr_direct_v6_5_fused(packed_w, v, v_i8, inv_scale, out,
                                          n, k, num_blocks, k_dim, grid)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(10):
            _module.rsr_direct_v6_5_fused(packed_w, v, v_i8, inv_scale, out,
                                          n, k, num_blocks, k_dim, grid)
        ev1.record()
        torch.cuda.synchronize()
        t = ev0.elapsed_time(ev1) / 10000.0
        if t < best_time:
            best_time = t
            best_grid = grid
    return best_grid


class RSRTernaryCudaV6_5Multiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int, k_dim: int = 32, grid_size: int = -1):
        assert M.shape[0] % k == 0
        assert k_dim in (16, 32)
        assert M.shape[0] % 4 == 0
        assert k in (2, 4, 6, 8, 10, 12)
        self.n = M.shape[0]
        self.k = k
        self.k_dim = k_dim
        self._requested_grid = grid_size  # -1 = auto-tune
        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        M_np = self.M.detach().cpu().to(torch.int8).numpy()
        packed = pack_weights_int2(M_np, self.n, self.k)
        self._packed = torch.from_numpy(packed).to(self.device)
        self._num_blocks = self.n // self.k
        self._out = torch.empty(self.n, dtype=torch.float32, device=self.device)
        self._v_i8_buf = torch.empty(self.n, dtype=torch.int8, device=self.device)
        self._inv_scale_buf = torch.empty(1, dtype=torch.float32, device=self.device)

        if self._requested_grid == -1:
            self._grid_size = _auto_tune_grid(
                self._packed, self._v_i8_buf, self._inv_scale_buf, self._out,
                self.n, self.k, self._num_blocks, self.k_dim, self.device)
        else:
            self._grid_size = self._requested_grid

        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_gpu = v.to(self.device) if v.device != self.device else v
        _module.rsr_direct_v6_5_fused(
            self._packed,
            v_gpu.contiguous(),
            self._v_i8_buf,
            self._inv_scale_buf,
            self._out,
            self.n, self.k, self._num_blocks, self.k_dim, self._grid_size,
        )
        return self._out
