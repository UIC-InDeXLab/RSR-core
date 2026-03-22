"""
RSR ternary CUDA v6.3 — v6.0 with wider loads (K_per_loop=32) + __ldg.

Processes 32 columns per loop iteration (2× int4 input, uint2 weight),
halving loop count and reducing decode overhead.
"""

import numpy as np
import torch

from multiplier.base import Multiplier
from ._jit_build import load_kernel
from .rsr_cuda_v6_0 import pack_weights_int2

_module = load_kernel("rsr_ternary_cuda_v6_3", "rsr_ternary_v6_3.cu")


class RSRTernaryCudaV6_3Multiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int, k_dim: int = 32):
        assert M.shape[0] % k == 0
        assert k_dim in (8, 16, 32)
        assert M.shape[0] % 4 == 0
        assert k in (2, 4, 6, 8, 10, 12)
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
        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_gpu = v.to(self.device) if v.device != self.device else v
        _module.rsr_direct_v6_3_fused(
            self._packed,
            v_gpu.contiguous(),
            self._v_i8_buf,
            self._inv_scale_buf,
            self._out,
            self.n, self.k, self._num_blocks, self.k_dim,
        )
        return self._out
