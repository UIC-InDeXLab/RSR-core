"""
RSR ternary CUDA v10.1 — v6.5 with a 64-column inner loop.

Processes 64 columns per loop iteration instead of 32 so the kernel spends
less time on loop control and decode overhead when n is large.
"""

import torch

from multiplier.base import Multiplier
from ._jit_build import load_kernel
from .rsr_cuda_v6_0 import pack_weights_int2
from .rsr_cuda_v6_5 import _auto_tune_grid

_module = load_kernel("rsr_ternary_cuda_v10_1", "rsr_ternary_v10_1.cu")


class RSRTernaryCudaV10_1Multiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int, k_dim: int = 32):
        assert M.shape[0] % k == 0
        assert M.shape[0] % 4 == 0
        assert k in (2, 4, 6, 8, 10, 12)
        assert k_dim in (16, 32)

        self.n = M.shape[0]
        self.k = k
        self.k_dim = k_dim
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
        self._grid_size = _auto_tune_grid(
            self._packed,
            self._v_i8_buf,
            self._inv_scale_buf,
            self._out,
            self.n,
            self.k,
            self._num_blocks,
            self.k_dim,
            self.device,
        )
        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_gpu = v if v.device == self.device and v.dtype == torch.float32 else v.to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        _module.rsr_direct_v10_1_fused(
            self._packed,
            v_gpu.contiguous(),
            self._v_i8_buf,
            self._inv_scale_buf,
            self._out,
            self.n,
            self.k,
            self._num_blocks,
            self.k_dim,
            self._grid_size,
        )
        return self._out
