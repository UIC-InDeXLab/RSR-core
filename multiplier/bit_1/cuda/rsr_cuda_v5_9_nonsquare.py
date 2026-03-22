"""
RSR CUDA v5.9 for non-square binary matrices (n_rows x n_cols).

Uses the non-square preprocessing kernel and reuses the v5.9 CUDA inference
kernel by passing n_cols as the permutation stride parameter.
Supports padding so that n_rows need not be divisible by k.
"""

import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

from ..base import Multiplier
from ._prep_cuda_nonsquare import prep_nonsquare_on_cpu_move_to_cuda

# Always set TORCH_CUDA_ARCH_LIST
if torch.cuda.is_available():
    _major, _minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_major}.{_minor}"

_kernel_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cuda")

_module = load(
    name="rsr_cuda_v5_9",
    sources=[os.path.join(_kernel_dir, "rsr_v5_9.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class RSRCudaV5_9NonSquareMultiplier(Multiplier):
    """Binary matrix-vector multiply for non-square M (n_rows x n_cols).

    M has entries in {0, 1}. Pads n_rows up to the nearest multiple of k.
    v must have length n_cols, output has length n_rows.
    """

    def __init__(self, M: torch.Tensor, k: int):
        assert M.ndim == 2, "M must be a 2D tensor"
        assert k > 0, "k must be positive"

        self.n_rows_orig = M.shape[0]
        self.n_cols = M.shape[1]
        self.k = k
        assert self.n_cols <= 65535, "v5.9 requires n_cols <= 65535 for uint16 perms"

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

        data = prep_nonsquare_on_cpu_move_to_cuda(M_padded, n_rows, n_cols, k, self.device)
        self._block_meta = data["block_meta"]
        self._num_blocks = data["num_blocks"]

        ge_cpu = data["group_ends"].cpu()
        bm_cpu = self._block_meta.cpu()
        so_cpu = data["scatter_offsets"].cpu()
        sr_cpu = data["scatter_rows"].cpu()

        group_starts_cpu = _module.compute_group_starts(
            ge_cpu, bm_cpu, self._num_blocks
        )
        group_masks_cpu = _module.compute_group_masks(
            so_cpu, sr_cpu
        )

        # Sort perms inside each group for gather locality
        perms_cpu = data["perms"].cpu().numpy()
        gs_np = group_starts_cpu.numpy()
        ge_np = ge_cpu.numpy()
        bm_np = bm_cpu.numpy()
        for b in range(self._num_blocks):
            g_off = int(bm_np[b * 2])
            n_groups = int(bm_np[b * 2 + 1])
            base = b * n_cols
            for g in range(n_groups):
                gg = g_off + g
                start = int(gs_np[gg])
                end = int(ge_np[gg])
                perms_cpu[base + start : base + end].sort()

        perms_u16 = np.asarray(perms_cpu, dtype=np.uint16)
        self._perms_u16 = torch.from_numpy(perms_u16.copy()).to(self.device)

        self._group_packed = _module.pack_group_metadata(
            group_starts_cpu, ge_cpu, group_masks_cpu
        ).to(self.device)
        self._out = torch.empty(n_rows, dtype=torch.float32, device=self.device)

        del self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        assert v.shape[0] == self.n_cols, (
            f"Expected vector length {self.n_cols}, got {v.shape[0]}"
        )

        v_gpu = v.to(self.device) if v.device != self.device else v

        # Reuse v5.9 kernel: pass n_cols as 'n' (perm stride)
        _module.rsr_gemv_v5_9(
            self._perms_u16,
            self._group_packed,
            self._block_meta,
            v_gpu,
            self._out,
            self.n_cols,
            self.k,
            self._num_blocks,
        )

        # Trim padded rows
        if self.row_pad > 0:
            return self._out[: self.n_rows_orig]
        return self._out
