"""
Ternary RSR v1.5 — C kernel with AVX2 gather + aggregate.

Same preprocessing as v1.4, but inference uses AVX2 vgatherdps for 8-wide
vectorized aggregation of permuted input elements.
"""

import ctypes
import os

import torch

from .rsr_v1_4 import RSRTernaryV1_4Multiplier

_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1_58", "cpu"
)

_lib = ctypes.CDLL(os.path.join(_DIR, "rsr_ternary_v1_5.so"))
_lib.rsr_ternary_gemv_v1_5.restype = None
_lib.rsr_ternary_gemv_v1_5.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # perms
    ctypes.POINTER(ctypes.c_int32),   # group_ends
    ctypes.POINTER(ctypes.c_int32),   # scatter_offsets
    ctypes.POINTER(ctypes.c_int8),    # scatter_rows
    ctypes.POINTER(ctypes.c_int8),    # scatter_signs
    ctypes.POINTER(ctypes.c_int32),   # block_meta
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.c_int,                     # n
    ctypes.c_int,                     # k
    ctypes.c_int,                     # num_blocks
]


class RSRTernaryV1_5Multiplier(RSRTernaryV1_4Multiplier):
    """Ternary RSR v1.5: AVX2-accelerated inference, same prep as v1.4."""

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

        _lib.rsr_ternary_gemv_v1_5(
            self._perms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._group_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._scatter_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._scatter_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self._scatter_signs.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self._block_meta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, self.k, self._num_blocks,
        )

        if v.device.type == "cpu":
            return out_cpu
        return out_cpu.to(v.device)
