"""
T-MAC-style ternary GEMV baseline (LUT-based, following microsoft/T-MAC).

Core idea: group 2 ternary weights per 4-bit nibble, build 16-entry LUT
per activation pair, use _mm256_shuffle_epi8 (pshufb) for 32 parallel lookups.
"""

import ctypes
import os

import numpy as np
import torch

from multiplier.base import Multiplier

_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1_58", "cpu"
)

_lib = ctypes.CDLL(os.path.join(_DIR, "tmac_ternary.so"))

_lib.tmac_ternary_pack.restype = None
_lib.tmac_ternary_pack.argtypes = [
    ctypes.POINTER(ctypes.c_int8),    # M
    ctypes.POINTER(ctypes.c_uint8),   # packed
    ctypes.c_int,                     # n
]

_lib.tmac_ternary_build_lut.restype = None
_lib.tmac_ternary_build_lut.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_int8),    # qlut
    ctypes.POINTER(ctypes.c_float),   # lut_scale
    ctypes.c_int,                     # n
]

_lib.tmac_ternary_gemv.restype = None
_lib.tmac_ternary_gemv.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # packed
    ctypes.POINTER(ctypes.c_int8),    # qlut
    ctypes.c_float,                   # lut_scale
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.c_int,                     # n
]


class TMACTernaryMultiplier(Multiplier):
    """T-MAC-style ternary GEMV: LUT + pshufb, groups of 2 ternary weights."""

    def __init__(self, M: torch.Tensor):
        n = M.shape[0]
        assert M.shape[1] == n, "Matrix must be square"
        self.n = n
        super().__init__(M)
        self.prep()

    def prep(self):
        n = self.n
        M_np = self.M.cpu().to(torch.int8).numpy().copy()

        n_groups = (n + 1) // 2
        row_bytes = (n + 1) // 2
        self._packed = np.zeros(n_groups * row_bytes, dtype=np.uint8)

        _lib.tmac_ternary_pack(
            M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self._packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n,
        )

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        n = self.n
        v_np = v.detach().cpu().float().numpy().copy()

        n_groups = (n + 1) // 2
        qlut = np.empty(n_groups * 16, dtype=np.int8)
        lut_scale = np.empty(1, dtype=np.float32)

        _lib.tmac_ternary_build_lut(
            v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            qlut.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            lut_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )

        out = np.empty(n, dtype=np.float32)
        _lib.tmac_ternary_gemv(
            self._packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            qlut.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            float(lut_scale[0]),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )

        return torch.from_numpy(out).to(v.device)
