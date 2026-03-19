"""
RSR CUDA Adaptive — supports any positive k by zero-padding.

Wraps RSRCudaV5_10Multiplier internally, padding M to the nearest multiple of k.
"""

import torch

from ..base import Multiplier
from .rsr_cuda_v5_10 import RSRCudaV5_10Multiplier


class RSRCudaAdaptiveMultiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int):
        assert k > 0, "k must be positive"
        assert M.ndim == 2 and M.shape[0] == M.shape[1], "M must be square 2D"

        self.n = M.shape[0]
        self.k = k
        self.n_padded = ((self.n + k - 1) // k) * k
        self.pad = self.n_padded - self.n
        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        if self.pad == 0:
            M_padded = self.M
        else:
            M_padded = torch.zeros(
                (self.n_padded, self.n_padded),
                dtype=self.M.dtype,
                device=self.M.device,
            )
            M_padded[: self.n, : self.n] = self.M

        self._inner = RSRCudaV5_10Multiplier(M_padded, self.k)
        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_gpu = v.to(self.device) if v.device != self.device else v

        if self.pad == 0:
            return self._inner(v_gpu)

        v_padded = torch.zeros(self.n_padded, dtype=v.dtype, device=self.device)
        v_padded[: self.n] = v_gpu
        return self._inner(v_padded)[: self.n]
