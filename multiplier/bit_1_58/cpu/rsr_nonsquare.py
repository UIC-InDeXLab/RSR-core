"""
Ternary RSR multiplier for non-square matrices (n_rows x n_cols).

Switches between v3.1 and v3.3 inference kernels based on matrix size:
  - n_cols >= 4096  →  v3.3 (bitmask scatter, lower bandwidth at scale)
  - n_cols <  4096  →  v3.1 (direct scatter, lighter metadata)

Both kernels require n_cols <= 65535 (uint16 addressing).
v3.3 additionally requires k <= 16 for bitmask scatter.
Supports square matrices as a special case (n_rows == n_cols).
"""

import ctypes
import os

import numpy as np
import torch

from multiplier.base import Multiplier
from ._rsr_v3_common import (
    INT8_PTR,
    INT32_PTR,
    UINT16_PTR,
    ensure_cpu_float32_contiguous,
    tensor_float_ptr,
)

_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1_58", "cpu"
)

# Non-square preprocessing kernel
_prep_lib = ctypes.CDLL(os.path.join(_DIR, "rsr_ternary_prep_nonsquare.so"))
_prep_lib.rsr_ternary_prep_nonsquare.restype = None
_prep_lib.rsr_ternary_prep_nonsquare.argtypes = [
    ctypes.POINTER(ctypes.c_int8),    # M
    ctypes.c_int,                     # n_rows
    ctypes.c_int,                     # n_cols
    ctypes.c_int,                     # k
    ctypes.POINTER(ctypes.c_int32),   # perms
    ctypes.POINTER(ctypes.c_int32),   # group_ends
    ctypes.POINTER(ctypes.c_int32),   # scatter_offsets
    ctypes.POINTER(ctypes.c_int8),    # scatter_rows
    ctypes.POINTER(ctypes.c_int8),    # scatter_signs
    ctypes.POINTER(ctypes.c_int32),   # block_meta
    ctypes.POINTER(ctypes.c_int32),   # out_sizes
]

# v3.3 inference kernel (bitmask scatter)
_v33_lib = ctypes.CDLL(os.path.join(_DIR, "rsr_ternary_v3_3.so"))
_v33_lib.rsr_ternary_gemv_v3_3.restype = None
_v33_lib.rsr_ternary_gemv_v3_3.argtypes = [
    UINT16_PTR, UINT16_PTR, UINT16_PTR, UINT16_PTR, INT32_PTR,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]

# v3.1 inference kernel (direct scatter)
_v31_lib = ctypes.CDLL(os.path.join(_DIR, "rsr_ternary_v3_1.so"))
_v31_lib.rsr_ternary_gemv_v3_1.restype = None
_v31_lib.rsr_ternary_gemv_v3_1.argtypes = [
    UINT16_PTR, UINT16_PTR, INT32_PTR, INT8_PTR, INT8_PTR, INT32_PTR,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]

# Threshold: use v3.3 for large n_cols, v3.1 for small
_V33_NCOLS_THRESHOLD = 4096


class RSRTernaryNonSquareMultiplier(Multiplier):
    """Ternary matrix-vector multiply for M (n_rows x n_cols).

    Automatically selects v3.1 or v3.3 kernel based on n_cols.
    Pads n_rows up to the nearest multiple of k.
    """

    def __init__(self, M: torch.Tensor, k: int):
        assert M.ndim == 2, "M must be a 2D tensor"
        assert k > 0, "k must be positive"
        assert k <= 12, f"k={k} exceeds max 12 for ternary counting-sort (4^k buckets)"

        self.n_rows_orig = M.shape[0]
        self.n_cols = M.shape[1]
        self.k = k

        assert self.n_cols <= np.iinfo(np.uint16).max, "requires n_cols <= 65535"

        # Select kernel: v3.3 for larger matrices, v3.1 for smaller
        self._use_v33 = (self.n_cols >= _V33_NCOLS_THRESHOLD and k <= 16)

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

        M_np = M_padded.cpu().to(torch.int8).numpy().copy()

        num_buckets = 1 << (2 * k)
        max_groups = num_blocks * min(num_buckets, n_cols)
        max_scatter = max_groups * 2 * k

        perms = np.empty(num_blocks * n_cols, dtype=np.int32)
        group_ends = np.empty(max_groups, dtype=np.int32)
        scatter_offsets = np.empty(max_groups + 1, dtype=np.int32)
        scatter_rows = np.empty(max_scatter, dtype=np.int8)
        scatter_signs = np.empty(max_scatter, dtype=np.int8)
        block_meta = np.empty(2 * num_blocks, dtype=np.int32)
        out_sizes = np.empty(2, dtype=np.int32)

        _prep_lib.rsr_ternary_prep_nonsquare(
            M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            n_rows, n_cols, k,
            perms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            group_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            scatter_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            scatter_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            scatter_signs.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            block_meta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )

        total_groups = int(out_sizes[0])
        total_scatter = int(out_sizes[1])

        self._block_meta = block_meta.copy()
        self._num_blocks = num_blocks

        # 16-bit permutation indices (shared by both kernels)
        self._perms_u16 = perms.astype(np.uint16, copy=True)
        self._group_ends_u16 = group_ends[:total_groups].astype(np.uint16, copy=True)

        self._perms_u16_ptr = self._perms_u16.ctypes.data_as(UINT16_PTR)
        self._group_ends_u16_ptr = self._group_ends_u16.ctypes.data_as(UINT16_PTR)
        self._block_meta_ptr = self._block_meta.ctypes.data_as(INT32_PTR)

        if self._use_v33:
            # v3.3: build bitmask representation
            pos_masks = np.zeros(total_groups, dtype=np.uint16)
            neg_masks = np.zeros(total_groups, dtype=np.uint16)
            soff = scatter_offsets[:total_groups + 1]
            for g in range(total_groups):
                start = int(soff[g])
                end = int(soff[g + 1])
                pos_mask = 0
                neg_mask = 0
                for s in range(start, end):
                    bit = 1 << int(scatter_rows[s])
                    if int(scatter_signs[s]) > 0:
                        pos_mask |= bit
                    else:
                        neg_mask |= bit
                pos_masks[g] = pos_mask
                neg_masks[g] = neg_mask

            self._pos_masks = pos_masks
            self._neg_masks = neg_masks
            self._pos_masks_ptr = self._pos_masks.ctypes.data_as(UINT16_PTR)
            self._neg_masks_ptr = self._neg_masks.ctypes.data_as(UINT16_PTR)
        else:
            # v3.1: keep scatter arrays
            self._scatter_offsets = scatter_offsets[:total_groups + 1].copy()
            self._scatter_rows = scatter_rows[:total_scatter].copy()
            self._scatter_signs = scatter_signs[:total_scatter].copy()
            self._scatter_offsets_ptr = self._scatter_offsets.ctypes.data_as(INT32_PTR)
            self._scatter_rows_ptr = self._scatter_rows.ctypes.data_as(INT8_PTR)
            self._scatter_signs_ptr = self._scatter_signs.ctypes.data_as(INT8_PTR)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        assert v.shape[0] == self.n_cols, (
            f"Expected vector length {self.n_cols}, got {v.shape[0]}"
        )

        v_cpu = ensure_cpu_float32_contiguous(v)
        out_cpu = torch.empty(self.n_rows_padded, dtype=torch.float32)

        if self._use_v33:
            _v33_lib.rsr_ternary_gemv_v3_3(
                self._perms_u16_ptr,
                self._group_ends_u16_ptr,
                self._pos_masks_ptr,
                self._neg_masks_ptr,
                self._block_meta_ptr,
                tensor_float_ptr(v_cpu),
                tensor_float_ptr(out_cpu),
                self.n_cols,
                self.k,
                self._num_blocks,
            )
        else:
            _v31_lib.rsr_ternary_gemv_v3_1(
                self._perms_u16_ptr,
                self._group_ends_u16_ptr,
                self._scatter_offsets_ptr,
                self._scatter_rows_ptr,
                self._scatter_signs_ptr,
                self._block_meta_ptr,
                tensor_float_ptr(v_cpu),
                tensor_float_ptr(out_cpu),
                self.n_cols,
                self.k,
                self._num_blocks,
            )

        result = out_cpu[: self.n_rows_orig]

        if v.device.type == "cpu":
            return result
        return result.to(v.device)
