"""
BitNet.cpp-style ternary GEMV baseline.

TQ2_0-style 2-bit packing with conditional add/sub kernel.
No multiplications in the inner loop — only additions and subtractions.
"""

import ctypes
import os

import numpy as np
import torch

from multiplier.base import Multiplier

_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1_58", "cpu"
)

_lib = ctypes.CDLL(os.path.join(_DIR, "bitnet_ternary.so"))

_lib.bitnet_ternary_pack.restype = None
_lib.bitnet_ternary_pack.argtypes = [
    ctypes.POINTER(ctypes.c_int8),    # M
    ctypes.POINTER(ctypes.c_uint8),   # packed
    ctypes.c_int,                     # n
]

_lib.bitnet_ternary_gemv.restype = None
_lib.bitnet_ternary_gemv.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # packed
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.c_int,                     # n
]


class BitNetTernaryMultiplier(Multiplier):
    """BitNet.cpp-style ternary GEMV: 2-bit packed, conditional add/sub."""

    def __init__(self, M: torch.Tensor):
        n = M.shape[0]
        assert M.shape[1] == n, "Matrix must be square"
        self.n = n
        super().__init__(M)
        self.prep()

    def prep(self):
        n = self.n
        M_np = self.M.cpu().to(torch.int8).numpy().copy()

        cols_packed = (n + 3) // 4
        self._packed = np.empty(n * cols_packed, dtype=np.uint8)

        _lib.bitnet_ternary_pack(
            M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self._packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n,
        )

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

        _lib.bitnet_ternary_gemv(
            self._packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )

        if v.device.type == "cpu":
            return out_cpu
        return out_cpu.to(v.device)
