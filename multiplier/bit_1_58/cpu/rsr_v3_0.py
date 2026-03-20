"""Ternary RSR v3.0: persistent scratch two-pass kernel."""

import ctypes
import os

import torch

from ._rsr_v3_common import INT8_PTR, INT32_PTR, ensure_cpu_float32_contiguous, tensor_float_ptr
from .rsr_v1_4 import RSRTernaryV1_4Multiplier

_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1_58", "cpu"
)

_lib = ctypes.CDLL(os.path.join(_DIR, "rsr_ternary_v3_0.so"))
_lib.rsr_ternary_gemv_v3_0.restype = None
_lib.rsr_ternary_gemv_v3_0.argtypes = [
    INT32_PTR,
    INT32_PTR,
    INT32_PTR,
    INT8_PTR,
    INT8_PTR,
    INT32_PTR,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


class RSRTernaryV3_0Multiplier(RSRTernaryV1_4Multiplier):
    """Two-pass ternary RSR with a persistent thread-local scratch buffer."""

    def prep(self):
        super().prep()
        self._perms_ptr = self._perms.ctypes.data_as(INT32_PTR)
        self._group_ends_ptr = self._group_ends.ctypes.data_as(INT32_PTR)
        self._scatter_offsets_ptr = self._scatter_offsets.ctypes.data_as(INT32_PTR)
        self._scatter_rows_ptr = self._scatter_rows.ctypes.data_as(INT8_PTR)
        self._scatter_signs_ptr = self._scatter_signs.ctypes.data_as(INT8_PTR)
        self._block_meta_ptr = self._block_meta.ctypes.data_as(INT32_PTR)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_cpu = ensure_cpu_float32_contiguous(v)
        out_cpu = torch.empty(self.n, dtype=torch.float32)

        _lib.rsr_ternary_gemv_v3_0(
            self._perms_ptr,
            self._group_ends_ptr,
            self._scatter_offsets_ptr,
            self._scatter_rows_ptr,
            self._scatter_signs_ptr,
            self._block_meta_ptr,
            tensor_float_ptr(v_cpu),
            tensor_float_ptr(out_cpu),
            self.n,
            self.k,
            self._num_blocks,
        )

        if v.device.type == "cpu":
            return out_cpu
        return out_cpu.to(v.device)
