"""RSR multiplier v2.4 — static scheduling + aligned buffers."""

import ctypes
import os

import torch
import numpy as np

from .rsr_cpp import RSRCppMultiplier

_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cpu", "rsr_v2_4.so"
)
_lib = ctypes.CDLL(_LIB_PATH)

_lib.rsr_gemv_v2_4.restype = None
_lib.rsr_gemv_v2_4.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


class RSRCppV2_4Multiplier(RSRCppMultiplier):
    """RSR v2.4: static sched + aligned bufs + no false sharing."""

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

        _lib.rsr_gemv_v2_4(
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
