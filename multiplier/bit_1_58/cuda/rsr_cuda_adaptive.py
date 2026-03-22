"""
RSR ternary CUDA adaptive — auto-selects best kernel based on n and k.

Selection strategy:
  - Uses v6.3 for all cases.
  - Non-divisible n: pads matrix to next multiple of lcm(effective_k, 32).
    The v6.3 kernel uses int4 (16-byte) loads on the input vector and
    uint32_t loads on packed weights (row stride = n/4 bytes), so n must be
    divisible by 32 to keep all memory accesses aligned.
"""

import math

import torch

from multiplier.base import Multiplier


class RSRTernaryCudaAdaptiveMultiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        self.device = torch.device("cuda")
        super().__init__(M)

        effective_k = min(k, 12)
        # v6.3 needs n divisible by both k and 32 (for aligned int4/uint32_t loads)
        align = math.lcm(effective_k, 32)

        if self.n % align != 0:
            padded_n = ((self.n + align - 1) // align) * align
            M_padded = torch.zeros(
                padded_n, padded_n, dtype=self.M.dtype, device=self.M.device
            )
            M_padded[: self.n, : self.n] = self.M
            self._padded = True
            self._padded_n = padded_n
            from .rsr_cuda_v6_3 import RSRTernaryCudaV6_3Multiplier

            self._inner = RSRTernaryCudaV6_3Multiplier(M_padded, effective_k)
        else:
            self._padded = False
            from .rsr_cuda_v6_3 import RSRTernaryCudaV6_3Multiplier

            self._inner = RSRTernaryCudaV6_3Multiplier(self.M, effective_k)

        del self.M

    def prep(self):
        pass

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        if self._padded:
            v_padded = torch.zeros(self._padded_n, dtype=v.dtype, device=v.device)
            v_padded[: self.n] = v.to(v_padded.device)
            result = self._inner(v_padded)
            return result[: self.n]
        return self._inner(v)
