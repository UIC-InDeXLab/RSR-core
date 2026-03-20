"""
Ternary RSR adaptive — supports any positive k by zero-padding M.

Wraps the fastest compatible ternary C kernel internally.
"""

import torch

from multiplier.base import Multiplier
from .rsr_v3_1 import RSRTernaryV3_1Multiplier
from .rsr_v2_0 import RSRTernaryV2_0Multiplier


class RSRTernaryAdaptiveMultiplier(Multiplier):
    """Ternary RSR that supports any positive k via zero-padding."""

    def __init__(self, M: torch.Tensor, k: int):
        assert k > 0, "k must be positive"
        assert M.ndim == 2 and M.shape[0] == M.shape[1], "Matrix must be square"

        self.n = M.shape[0]
        self.k = k
        self.n_padded = ((self.n + k - 1) // k) * k
        self.pad = self.n_padded - self.n
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

        if self.n_padded <= 65535:
            self._inner = RSRTernaryV3_1Multiplier(M_padded, self.k)
        else:
            self._inner = RSRTernaryV2_0Multiplier(M_padded, self.k)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        if self.pad == 0:
            v_padded = v
        else:
            v_padded = torch.zeros(self.n_padded, dtype=v.dtype, device=v.device)
            v_padded[: self.n] = v

        out_padded = self._inner(v_padded)
        return out_padded[: self.n]
