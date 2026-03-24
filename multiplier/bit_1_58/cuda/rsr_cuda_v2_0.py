"""RSR Ternary CUDA v2.0: compact metadata + static-k kernel."""

import torch

from multiplier.base import Multiplier
from ._jit_build import load_kernel
from ._prep_v2_common import prep_compact_u64


class RSRTernaryCudaV2_0Multiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int):
        assert M.ndim == 2
        assert 0 < k <= 16
        self.n_rows_orig = M.shape[0]
        self.n_cols = M.shape[1]
        self.k = k
        assert self.n_cols <= 65535
        self.n_rows_padded = ((self.n_rows_orig + k - 1) // k) * k
        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        self._module = load_kernel("rsr_ternary_cuda_v2_0", "rsr_ternary_v2_0.cu")
        self._perms_u16, self._group_packed_u64, self._block_meta, self._num_blocks = prep_compact_u64(
            self.M,
            self.n_rows_orig,
            self.n_cols,
            self.k,
            self.n_rows_padded,
            self.device,
        )
        self._out = torch.empty(self.n_rows_padded, dtype=torch.float32, device=self.device)
        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_gpu = v.to(self.device, dtype=torch.float32) if v.device != self.device or v.dtype != torch.float32 else v
        self._module.rsr_ternary_gemv_v2_0(
            self._perms_u16,
            self._group_packed_u64,
            self._block_meta,
            v_gpu,
            self._out,
            self.n_cols,
            self.k,
            self._num_blocks,
        )
        return self._out[: self.n_rows_orig]
