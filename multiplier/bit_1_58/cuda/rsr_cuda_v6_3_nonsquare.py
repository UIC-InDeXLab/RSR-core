"""
RSR ternary CUDA v6.3 for non-square matrices (n_rows x n_cols).

Reuses the v6.3 CUDA kernel (INT2 weights + INT8 input + dp4a) by passing
n_cols as the column dimension parameter. Only the weight packing needs a
non-square variant. Supports padding so that n_rows need not be divisible by k.
"""

import numpy as np
import torch

from multiplier.base import Multiplier
from ._jit_build import load_kernel

_module = load_kernel("rsr_ternary_cuda_v6_3", "rsr_ternary_v6_3.cu")


def _pack_weights_int2_nonsquare(M_np: np.ndarray, n_rows: int, n_cols: int, k: int) -> np.ndarray:
    """Pack non-square ternary matrix as INT2: 4 consecutive columns per byte.

    Encoding: -1 -> 1, 0 -> 2, +1 -> 3  (decoded in kernel by subtracting 2)
    Output shape: [num_blocks * k, n_cols // 4] uint8
    """
    num_blocks = n_rows // k
    M_r = M_np.reshape(num_blocks, k, n_cols)
    encoded = (M_r.astype(np.int8) + 2).astype(np.uint8)
    enc4 = encoded.reshape(num_blocks, k, n_cols // 4, 4)
    packed = (enc4[:, :, :, 0]
              | (enc4[:, :, :, 1] << 2)
              | (enc4[:, :, :, 2] << 4)
              | (enc4[:, :, :, 3] << 6)).astype(np.uint8)
    return packed.reshape(num_blocks * k, n_cols // 4)


class RSRTernaryCudaV6_3NonSquareMultiplier(Multiplier):
    """Ternary matrix-vector multiply for non-square M (n_rows x n_cols).

    M has entries in {-1, 0, +1}. Pads n_rows up to the nearest multiple of k.
    n_cols must be divisible by 4. v must have length n_cols, output has length n_rows.
    """

    def __init__(self, M: torch.Tensor, k: int, k_dim: int = 32):
        assert M.ndim == 2, "M must be a 2D tensor"
        assert k in (2, 4, 6, 8, 10, 12), f"k={k} not supported"
        assert k_dim in (8, 16, 32), "k_dim must be 8, 16, or 32"

        self.n_rows_orig = M.shape[0]
        self.n_cols = M.shape[1]
        self.k = k
        self.k_dim = k_dim
        assert self.n_cols % 4 == 0, "n_cols must be divisible by 4"

        # Pad rows to a multiple of k
        self.n_rows_padded = ((self.n_rows_orig + k - 1) // k) * k
        self.row_pad = self.n_rows_padded - self.n_rows_orig

        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        k = self.k
        n_rows = self.n_rows_padded
        n_cols = self.n_cols

        # Pad matrix rows if needed
        if self.row_pad > 0:
            M_padded = torch.zeros(
                (n_rows, n_cols), dtype=self.M.dtype, device=self.M.device
            )
            M_padded[: self.n_rows_orig, :] = self.M
        else:
            M_padded = self.M

        M_np = M_padded.detach().cpu().to(torch.int8).numpy()
        packed = _pack_weights_int2_nonsquare(M_np, n_rows, n_cols, k)
        self._packed = torch.from_numpy(packed).to(self.device)
        self._num_blocks = n_rows // k
        self._out = torch.empty(n_rows, dtype=torch.float32, device=self.device)
        self._v_i8_buf = torch.empty(n_cols, dtype=torch.int8, device=self.device)
        self._inv_scale_buf = torch.empty(1, dtype=torch.float32, device=self.device)
        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        assert v.shape[0] == self.n_cols, (
            f"Expected vector length {self.n_cols}, got {v.shape[0]}"
        )

        v_gpu = v.to(self.device) if v.device != self.device else v

        # Reuse v6.3 kernel: pass n_cols as 'n' (column dimension)
        _module.rsr_direct_v6_3_fused(
            self._packed,
            v_gpu.contiguous(),
            self._v_i8_buf,
            self._inv_scale_buf,
            self._out,
            self.n_cols, self.k, self._num_blocks, self.k_dim,
        )

        # Trim padded rows
        if self.row_pad > 0:
            return self._out[: self.n_rows_orig]
        return self._out
