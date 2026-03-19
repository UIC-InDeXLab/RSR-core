"""
RSR CUDA v5.10 — hybrid dispatcher tuned for large n.

Dispatch policy:
  - k == 8 and n <= 4096: use v5.7
  - k == 16 and n <= 8192: use v5.8
  - otherwise: use v5.9
"""

import torch

from ..base import Multiplier
from .rsr_cuda_v5_7 import RSRCudaV5_7Multiplier
from .rsr_cuda_v5_8 import RSRCudaV5_8Multiplier
from .rsr_cuda_v5_9 import RSRCudaV5_9Multiplier


class RSRCudaV5_10Multiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        assert self.n % k == 0
        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        if self.k == 8 and self.n <= 4096:
            self._impl = RSRCudaV5_7Multiplier(self.M, self.k)
        elif self.k == 16 and self.n <= 8192:
            self._impl = RSRCudaV5_8Multiplier(self.M, self.k)
        else:
            self._impl = RSRCudaV5_9Multiplier(self.M, self.k)

        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self._impl(v)
