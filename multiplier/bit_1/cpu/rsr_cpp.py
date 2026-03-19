"""
RSR multiplier backed by an optimized C kernel (kernels/cpu/rsr.c).

Preprocessing stays in Python/PyTorch (one-time cost).
Inference calls a fused C kernel that does gather + aggregate + scatter
in a single pass with AVX2 SIMD for the gather+sum phase.
"""

import ctypes
import os

import torch
import numpy as np

from ..base import Multiplier

_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cpu", "rsr.so"
)
_lib = ctypes.CDLL(_LIB_PATH)

_prep_lib = ctypes.CDLL(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cpu", "rsr_prep.so"
))
_prep_lib.rsr_prep_omp.restype = None
_prep_lib.rsr_prep_omp.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # M
    ctypes.c_int,                      # n
    ctypes.c_int,                      # k
    ctypes.POINTER(ctypes.c_int32),   # perms
    ctypes.POINTER(ctypes.c_int32),   # group_ends
    ctypes.POINTER(ctypes.c_int32),   # scatter_offsets
    ctypes.POINTER(ctypes.c_uint8),   # scatter_rows
    ctypes.POINTER(ctypes.c_int32),   # block_meta
    ctypes.POINTER(ctypes.c_int32),   # out_sizes
]

_lib.rsr_gemv.restype = None
_lib.rsr_gemv.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # perms
    ctypes.POINTER(ctypes.c_int32),   # group_ends
    ctypes.POINTER(ctypes.c_int32),   # scatter_offsets
    ctypes.POINTER(ctypes.c_uint8),   # scatter_rows
    ctypes.POINTER(ctypes.c_int32),   # block_meta
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.c_int,                      # n
    ctypes.c_int,                      # k
    ctypes.c_int,                      # num_blocks
]


class RSRCppMultiplier(Multiplier):
    """Binary matrix-vector multiply using RSR with a C kernel for inference."""

    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        assert self.n % k == 0, f"n={self.n} must be divisible by k={k}"
        assert k <= 20, f"k={k} exceeds max 20 for counting-sort prep kernel"
        super().__init__(M)
        self.prep()

    def prep(self):
        n, k = self.n, self.k
        num_blocks = n // k
        num_buckets = 1 << k

        M_np = self.M.cpu().numpy().astype(np.uint8).copy()

        max_groups = num_blocks * min(num_buckets, n)
        max_scatter = max_groups * k

        perms = np.empty(num_blocks * n, dtype=np.int32)
        group_ends = np.empty(max_groups, dtype=np.int32)
        scatter_offsets = np.empty(max_groups + 1, dtype=np.int32)
        scatter_rows = np.empty(max_scatter, dtype=np.uint8)
        block_meta = np.empty(2 * num_blocks, dtype=np.int32)
        out_sizes = np.empty(2, dtype=np.int32)

        _prep_lib.rsr_prep_omp(
            M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n, k,
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
        n = self.n
        v_detached = v.detach()
        if (
            v_detached.device.type == "cpu"
            and v_detached.dtype == torch.float32
            and v_detached.is_contiguous()
        ):
            v_cpu = v_detached
        else:
            v_cpu = v_detached.to(device="cpu", dtype=torch.float32).contiguous()

        out_cpu = torch.empty(n, dtype=torch.float32)
        v_np = v_cpu.numpy()
        out_np = out_cpu.numpy()

        _lib.rsr_gemv(
            self._perms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._group_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._scatter_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._scatter_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._block_meta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, self.k, self._num_blocks,
        )

        if v.device.type == "cpu":
            return out_cpu
        return out_cpu.to(v.device)
