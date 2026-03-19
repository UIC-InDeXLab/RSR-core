"""
BitNet.cpp-like I2_S baseline with official-style 2-bit coding.

This module mirrors BitNet.cpp's I2_S ideas more closely than bitnet.py:
  - Weight coding uses 0/1/2 slots for (-1, 0, +1)-style representation.
  - 4-row interleaved packing layout from quantize_i2_s (non-ACT path).
  - int8 activation quantization + SIMD int2xint8 dot products.
"""

import ctypes
import os

import numpy as np
import torch

from ..base import Multiplier

_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cpu", "bitnet_official.so"
)
_lib = ctypes.CDLL(_LIB_PATH)

_lib.bitnet_official_pack_weights.restype = None
_lib.bitnet_official_pack_weights.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # weights
    ctypes.POINTER(ctypes.c_uint8),   # packed
    ctypes.c_int,                     # n_rows
    ctypes.c_int,                     # n_cols
    ctypes.POINTER(ctypes.c_float),   # i2_scale_out
]

_lib.bitnet_official_quantize_activation.restype = ctypes.c_float
_lib.bitnet_official_quantize_activation.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_int8),    # qv
    ctypes.c_int,                     # n
]

_lib.bitnet_official_gemv.restype = None
_lib.bitnet_official_gemv.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # packed
    ctypes.POINTER(ctypes.c_int8),    # qv
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.c_int,                     # n_rows
    ctypes.c_int,                     # n_cols
    ctypes.c_float,                   # inv_act_scale
    ctypes.c_float,                   # i2_scale
]


class BitNetOfficialMultiplier(Multiplier):
    """BitNet.cpp-like I2_S matrix-vector multiply baseline."""

    def __init__(self, M: torch.Tensor):
        n = M.shape[0]
        assert M.shape[1] == n, "Matrix must be square"
        assert n % 4 == 0, f"n={n} must be divisible by 4 (I2_S row group size)"
        self.n = n
        super().__init__(M)
        self.prep()

    def prep(self):
        n = self.n
        M_np = self.M.detach().cpu().float().numpy().copy()
        self._packed = np.empty((n * n) // 4, dtype=np.uint8)
        scale = np.empty(1, dtype=np.float32)

        _lib.bitnet_official_pack_weights(
            M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n,
            n,
            scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        self._i2_scale = float(scale[0])

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        n = self.n
        v_np = v.detach().cpu().float().numpy().copy()
        qv = np.empty(n, dtype=np.int8)

        inv_act_scale = _lib.bitnet_official_quantize_activation(
            v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            qv.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            n,
        )

        out = np.empty(n, dtype=np.float32)
        _lib.bitnet_official_gemv(
            self._packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            qv.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            n,
            inv_act_scale,
            self._i2_scale,
        )

        return torch.from_numpy(out).to(v.device)
