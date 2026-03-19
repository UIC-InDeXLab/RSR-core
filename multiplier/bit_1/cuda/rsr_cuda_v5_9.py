"""
RSR CUDA v5.9 — packed metadata kernel with uint16 perms for large-n throughput.

Uses sorted perms within each group and stores them as uint16 (n <= 65535).
"""

import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

from ..base import Multiplier
from ._prep_cuda import prep_on_cpu_move_to_cuda

_kernel_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cuda")

_module = load(
    name="rsr_cuda_v5_9",
    sources=[os.path.join(_kernel_dir, "rsr_v5_9.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=True,  # see the JIT logs
)


class RSRCudaV5_9Multiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        assert self.n % k == 0
        assert self.n <= 65535, "v5.9 requires n <= 65535 for uint16 perms"
        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        data = prep_on_cpu_move_to_cuda(self.M, self.n, self.k, self.device)
        self._block_meta = data["block_meta"]
        self._num_blocks = data["num_blocks"]

        ge_cpu = data["group_ends"].cpu()
        bm_cpu = self._block_meta.cpu()
        so_cpu = data["scatter_offsets"].cpu()
        sr_cpu = data["scatter_rows"].cpu()

        group_starts_cpu = _module.compute_group_starts(  # type: ignore
            ge_cpu, bm_cpu, self._num_blocks
        )
        group_masks_cpu = _module.compute_group_masks(  # type: ignore
            so_cpu, sr_cpu
        )

        # Sort perms inside each group to improve gather locality.
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

        perms_u16 = np.asarray(perms_cpu, dtype=np.uint16)
        self._perms_u16 = torch.from_numpy(perms_u16.copy()).to(self.device)

        self._group_packed = _module.pack_group_metadata(  # type: ignore
            group_starts_cpu, ge_cpu, group_masks_cpu
        ).to(self.device)
        self._out = torch.empty(self.n, dtype=torch.float32, device=self.device)

        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_gpu = v.to(self.device) if v.device != self.device else v
        _module.rsr_gemv_v5_9(  # type: ignore
            self._perms_u16,
            self._group_packed,
            self._block_meta,
            v_gpu,
            self._out,
            self.n,
            self.k,
            self._num_blocks,
        )
        return self._out
