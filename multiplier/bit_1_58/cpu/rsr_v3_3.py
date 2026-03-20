"""Ternary RSR v3.3: direct gather with compact scatter bitmasks."""

import ctypes
import os

import numpy as np
import torch

from ._rsr_v3_common import INT32_PTR, UINT16_PTR, ensure_cpu_float32_contiguous, tensor_float_ptr
from .rsr_v1_4 import RSRTernaryV1_4Multiplier

_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1_58", "cpu"
)

_lib = ctypes.CDLL(os.path.join(_DIR, "rsr_ternary_v3_3.so"))
_lib.rsr_ternary_gemv_v3_3.restype = None
_lib.rsr_ternary_gemv_v3_3.argtypes = [
    UINT16_PTR,
    UINT16_PTR,
    UINT16_PTR,
    UINT16_PTR,
    INT32_PTR,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


class RSRTernaryV3_3Multiplier(RSRTernaryV1_4Multiplier):
    """Direct-gather ternary RSR with bitmask-encoded scatter metadata."""

    def prep(self):
        super().prep()
        assert self.n <= np.iinfo(np.uint16).max, "v3.3 requires n <= 65535"
        assert self.k <= 16, "v3.3 requires k <= 16"

        self._perms_u16 = self._perms.astype(np.uint16, copy=True)
        self._group_ends_u16 = self._group_ends.astype(np.uint16, copy=True)

        total_groups = len(self._group_ends)
        pos_masks = np.zeros(total_groups, dtype=np.uint16)
        neg_masks = np.zeros(total_groups, dtype=np.uint16)
        for g in range(total_groups):
            start = int(self._scatter_offsets[g])
            end = int(self._scatter_offsets[g + 1])
            pos_mask = 0
            neg_mask = 0
            for s in range(start, end):
                row = int(self._scatter_rows[s])
                bit = 1 << row
                if int(self._scatter_signs[s]) > 0:
                    pos_mask |= bit
                else:
                    neg_mask |= bit
            pos_masks[g] = pos_mask
            neg_masks[g] = neg_mask

        self._pos_masks = pos_masks
        self._neg_masks = neg_masks

        self._perms_u16_ptr = self._perms_u16.ctypes.data_as(UINT16_PTR)
        self._group_ends_u16_ptr = self._group_ends_u16.ctypes.data_as(UINT16_PTR)
        self._pos_masks_ptr = self._pos_masks.ctypes.data_as(UINT16_PTR)
        self._neg_masks_ptr = self._neg_masks.ctypes.data_as(UINT16_PTR)
        self._block_meta_ptr = self._block_meta.ctypes.data_as(INT32_PTR)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_cpu = ensure_cpu_float32_contiguous(v)
        out_cpu = torch.empty(self.n, dtype=torch.float32)

        _lib.rsr_ternary_gemv_v3_3(
            self._perms_u16_ptr,
            self._group_ends_u16_ptr,
            self._pos_masks_ptr,
            self._neg_masks_ptr,
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
