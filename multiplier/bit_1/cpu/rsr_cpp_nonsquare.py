"""
RSR multiplier for non-square binary matrices (n_rows x n_cols).

Uses a dedicated non-square preprocessing kernel and reuses the v4.2
inference kernel (which only needs n_cols as its column-stride parameter).
Supports padding so that n_rows need not be divisible by k.
"""

import ctypes
import os

import torch
import numpy as np

from ..base import Multiplier

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cpu"
)

_prep_lib = ctypes.CDLL(os.path.join(_KERNEL_DIR, "rsr_prep_nonsquare.so"))
_prep_lib.rsr_prep_nonsquare_omp.restype = None
_prep_lib.rsr_prep_nonsquare_omp.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # M
    ctypes.c_int,                      # n_rows
    ctypes.c_int,                      # n_cols
    ctypes.c_int,                      # k
    ctypes.POINTER(ctypes.c_int32),   # perms
    ctypes.POINTER(ctypes.c_int32),   # group_ends
    ctypes.POINTER(ctypes.c_int32),   # scatter_offsets
    ctypes.POINTER(ctypes.c_uint8),   # scatter_rows
    ctypes.POINTER(ctypes.c_int32),   # block_meta
    ctypes.POINTER(ctypes.c_int32),   # out_sizes
]

_v42_lib = ctypes.CDLL(os.path.join(_KERNEL_DIR, "rsr_v4_2.so"))
_v42_lib.rsr_gemv_v4_2.restype = None
_v42_lib.rsr_gemv_v4_2.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # perms
    ctypes.POINTER(ctypes.c_int32),   # group_ends
    ctypes.POINTER(ctypes.c_int32),   # scatter_offsets
    ctypes.POINTER(ctypes.c_uint8),   # scatter_rows
    ctypes.POINTER(ctypes.c_int32),   # block_meta
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.c_int,                      # n  (= n_cols for non-square)
    ctypes.c_int,                      # k
    ctypes.c_int,                      # num_blocks
]


class RSRCppNonSquareMultiplier(Multiplier):
    """Binary matrix-vector multiply for non-square M (n_rows x n_cols).

    Pads n_rows up to the nearest multiple of k if needed.
    v must have length n_cols, output has length n_rows.
    """

    def __init__(self, M: torch.Tensor, k: int):
        assert M.ndim == 2, "M must be a 2D tensor"
        assert k > 0, "k must be positive"
        assert k <= 20, f"k={k} exceeds max 20 for counting-sort prep kernel"

        self.n_rows_orig = M.shape[0]
        self.n_cols = M.shape[1]
        self.k = k

        # Pad rows to a multiple of k
        self.n_rows_padded = ((self.n_rows_orig + k - 1) // k) * k
        self.row_pad = self.n_rows_padded - self.n_rows_orig

        super().__init__(M)
        self.prep()

    def prep(self):
        k = self.k
        n_rows = self.n_rows_padded
        n_cols = self.n_cols
        num_blocks = n_rows // k

        # Pad matrix rows if needed
        if self.row_pad > 0:
            M_padded = torch.zeros(
                (n_rows, n_cols), dtype=self.M.dtype, device=self.M.device
            )
            M_padded[: self.n_rows_orig, :] = self.M
        else:
            M_padded = self.M

        M_np = M_padded.cpu().numpy().astype(np.uint8).copy()

        num_buckets = 1 << k
        max_groups = num_blocks * min(num_buckets, n_cols)
        max_scatter = max_groups * k

        perms = np.empty(num_blocks * n_cols, dtype=np.int32)
        group_ends = np.empty(max_groups, dtype=np.int32)
        scatter_offsets = np.empty(max_groups + 1, dtype=np.int32)
        scatter_rows = np.empty(max_scatter, dtype=np.uint8)
        block_meta = np.empty(2 * num_blocks, dtype=np.int32)
        out_sizes = np.empty(2, dtype=np.int32)

        _prep_lib.rsr_prep_nonsquare_omp(
            M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n_rows, n_cols, k,
            perms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            group_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            scatter_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            scatter_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            block_meta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )

        total_groups = int(out_sizes[0])
        total_scatter = int(out_sizes[1])

        self._perms = perms.copy()
        self._group_ends = group_ends[:total_groups].copy()
        self._scatter_offsets = scatter_offsets[:total_groups + 1].copy()
        self._scatter_rows = scatter_rows[:total_scatter].copy()
        self._block_meta = block_meta.copy()
        self._num_blocks = num_blocks

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        assert v.shape[0] == self.n_cols, (
            f"Expected vector length {self.n_cols}, got {v.shape[0]}"
        )

        v_detached = v.detach()
        if (
            v_detached.device.type == "cpu"
            and v_detached.dtype == torch.float32
            and v_detached.is_contiguous()
        ):
            v_cpu = v_detached
        else:
            v_cpu = v_detached.to(device="cpu", dtype=torch.float32).contiguous()

        n_rows = self.n_rows_padded
        out_cpu = torch.empty(n_rows, dtype=torch.float32)
        v_np = v_cpu.numpy()
        out_np = out_cpu.numpy()

        # Reuse v4.2 kernel: pass n_cols as 'n' (perm stride)
        _v42_lib.rsr_gemv_v4_2(
            self._perms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._group_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._scatter_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._scatter_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._block_meta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.n_cols, self.k, self._num_blocks,
        )

        # Trim padded rows
        result = out_cpu[: self.n_rows_orig]

        if v.device.type == "cpu":
            return result
        return result.to(v.device)
