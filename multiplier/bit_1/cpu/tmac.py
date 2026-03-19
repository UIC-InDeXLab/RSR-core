"""
T-MAC-style binary GEMV baseline (LUT-based, following microsoft/T-MAC).

For binary {0,1} weights: maps to T-MAC's {-1,+1} representation,
groups 4 weights per nibble (2^4=16-entry LUT), uses pshufb for lookups.
Corrects with: result = (tmac_result + sum(v)) / 2.
"""

import ctypes
import os

import numpy as np
import torch

from multiplier.base import Multiplier

_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cpu", "tmac_binary.so"
)

_lib = ctypes.CDLL(_LIB_PATH)

_lib.tmac_binary_pack.restype = None
_lib.tmac_binary_pack.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # M
    ctypes.POINTER(ctypes.c_uint8),   # packed
    ctypes.c_int,                     # n
]

_lib.tmac_binary_build_lut.restype = None
_lib.tmac_binary_build_lut.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_int8),    # qlut
    ctypes.POINTER(ctypes.c_float),   # lut_scale
    ctypes.c_int,                     # n
]

_lib.tmac_binary_gemv.restype = None
_lib.tmac_binary_gemv.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # packed
    ctypes.POINTER(ctypes.c_int8),    # qlut
    ctypes.c_float,                   # lut_scale
    ctypes.POINTER(ctypes.c_float),   # v (for sum_v)
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.c_int,                     # n
]


class TMACBinaryMultiplier(Multiplier):
    """T-MAC-style binary GEMV: LUT + pshufb, groups of 4 binary weights."""

    def __init__(self, M: torch.Tensor):
        n = M.shape[0]
        assert M.shape[1] == n, "Matrix must be square"
        self.n = n
        super().__init__(M)
        self.prep()

    def prep(self):
        n = self.n
        M_np = self.M.detach().cpu().float().numpy().copy()

        n_groups = (n + 3) // 4
        row_bytes = (n + 1) // 2
        self._packed = np.zeros(n_groups * row_bytes, dtype=np.uint8)

        _lib.tmac_binary_pack(
            M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n,
        )

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        n = self.n
        v_np = v.detach().cpu().float().numpy().copy()

        n_groups = (n + 3) // 4
        qlut = np.empty(n_groups * 16, dtype=np.int8)
        lut_scale = np.empty(1, dtype=np.float32)

        _lib.tmac_binary_build_lut(
            v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            qlut.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            lut_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )

        out = np.empty(n, dtype=np.float32)
        _lib.tmac_binary_gemv(
            self._packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            qlut.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            float(lut_scale[0]),
            v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )

        return torch.from_numpy(out).to(v.device)
