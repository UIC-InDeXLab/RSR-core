"""
RSR ternary CUDA v6.0 — INT2 weights + INT8 input + dp4a.

Borrows BitNet's full compute stack:
  - Weights packed as INT2 (4 per byte, sequential column order)
  - Input quantized to INT8 absmax (fused on GPU, no D2H sync)
  - __dp4a for INT8×INT8→INT32 (4 MACs per instruction)
  - Warp-level reduction, no shared memory

Parametric k_dim (threads per output row):
  k_dim=16 → THREADS = k*16; k=8 gives 128 threads (same as BitNet)
  k_dim=8  → THREADS = k*8;  k=8 gives  64 threads (2× BitNet occupancy)
  k_dim=32 → THREADS = k*32; k=8 gives 256 threads (baseline)
"""

import numpy as np
import torch

from multiplier.base import Multiplier
from ._jit_build import load_kernel

_module = load_kernel("rsr_ternary_cuda_v6_0", "rsr_ternary_v6_0.cu")


def pack_weights_int2(M_np: np.ndarray, n: int, k: int) -> np.ndarray:
    """Pack ternary matrix as INT2: 4 consecutive columns per byte.

    Encoding: -1 → 1, 0 → 2, +1 → 3  (decoded in kernel by subtracting 2)
    Output shape: [num_blocks * k, n // 4] uint8
    """
    num_blocks = n // k
    # Reshape: M_np (n, n) → (num_blocks, k, n)
    M_r = M_np.reshape(num_blocks, k, n)
    # Encode
    encoded = (M_r.astype(np.int8) + 2).astype(np.uint8)  # 1, 2, or 3
    # Pack 4 consecutive cols into 1 byte
    enc4 = encoded.reshape(num_blocks, k, n // 4, 4)
    packed = (enc4[:, :, :, 0]
              | (enc4[:, :, :, 1] << 2)
              | (enc4[:, :, :, 2] << 4)
              | (enc4[:, :, :, 3] << 6)).astype(np.uint8)
    return packed.reshape(num_blocks * k, n // 4)


class RSRTernaryCudaV6_0Multiplier(Multiplier):
    """v6.0: INT2 weights + INT8 input + dp4a, k_dim=16 by default."""

    def __init__(self, M: torch.Tensor, k: int, k_dim: int = 16):
        assert M.shape[0] % k == 0, f"n={M.shape[0]} not divisible by k={k}"
        assert k_dim in (8, 16, 32), "k_dim must be 8, 16, or 32"
        assert M.shape[0] % 4 == 0, "n must be divisible by 4"
        assert k in (2, 4, 6, 8, 10, 12), f"k={k} not supported"
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
        _module.rsr_direct_v6_0_fused(
            self._packed,
            v_gpu.contiguous(),
            self._v_i8_buf,
            self._inv_scale_buf,
            self._out,
            self.n, self.k, self._num_blocks, self.k_dim,
        )
        return self._out
