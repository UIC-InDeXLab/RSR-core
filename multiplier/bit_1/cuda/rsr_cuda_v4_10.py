"""
RSR CUDA v4.10 — uint16 perms + 8x unroll.

Building on v4.8:
  1. uint16_t perms → supports n up to 65535 (vs 32767 with int16).
  2. 8x unroll for better ILP on large groups.
  3. Adaptive thread count from v4.7.
  4. Sorted perms from v4.3.
"""

import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

from ..base import Multiplier
from ._prep_cuda import prep_on_cpu_move_to_cuda

_kernel_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cuda")

_module = load(
    name="rsr_cuda_v4_10",
    sources=[os.path.join(_kernel_dir, "rsr_v4_10.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


def _choose_threads(k: int) -> int:
    if k <= 2:
        return 128
    elif k <= 4:
        return 256
    else:
        return 512


class RSRCudaV4_10Multiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        assert self.n % k == 0
        assert self.n <= 65535, f"v4.10 requires n <= 65535, got n={self.n}"
        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        data = prep_on_cpu_move_to_cuda(self.M, self.n, self.k, self.device)
        self._group_ends = data["group_ends"]
        self._scatter_offsets = data["scatter_offsets"]
        self._scatter_rows = data["scatter_rows"]
        self._block_meta = data["block_meta"]
        self._num_blocks = data["num_blocks"]

        ge_cpu = data["group_ends"].cpu()
        bm_cpu = data["block_meta"].cpu()
        group_starts_cpu = _module.compute_group_starts(
            ge_cpu, bm_cpu, self._num_blocks
        )

        perms_cpu = data["perms"].cpu().numpy()
        gs_np = group_starts_cpu.numpy()
        ge_np = ge_cpu.numpy()
        bm_np = bm_cpu.numpy()

        for b in range(self._num_blocks):
            g_off = int(bm_np[b * 2])
            n_groups = int(bm_np[b * 2 + 1])
            base = b * self.n
            for g in range(n_groups):
                gg = g_off + g
                start = int(gs_np[gg])
                end = int(ge_np[gg])
                perms_cpu[base + start : base + end].sort()

        # Store as uint16 via torch.int16 (same bits, kernel casts to uint16_t)
        perms_i16 = perms_cpu.copy().astype(np.int16)
        self._perms = torch.from_numpy(perms_i16).to(self.device)
        self._group_starts = group_starts_cpu.to(self.device)
        self._threads = _choose_threads(self.k)
        self._out = torch.empty(self.n, dtype=torch.float32, device=self.device)

        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_gpu = v.to(self.device) if v.device != self.device else v
        _module.rsr_gemv_v4_10(
            self._perms, self._group_starts, self._group_ends,
            self._scatter_offsets, self._scatter_rows, self._block_meta,
            v_gpu, self._out, self.n, self.k, self._num_blocks, self._threads,
        )
        return self._out
