"""Runtime RSR multipliers that operate on preprocessed (saved) tensors.

These classes mirror the prep-time multipliers but skip the preprocessing
step — they load directly from safetensors artifacts produced by
``integrations.hf.model_prep``.

Classes:
    RSRPreprocessedMultiplier   — single-layer CPU GEMV (v3.1 or v3.3)
    RSRBatchMultiplier          — multi-layer batched GEMV (v3.3 only)
    RSRBatchContext             — orchestrates batch execution for layers
                                  sharing the same input vector
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np
import torch

from ._rsr_v3_common import (
    INT8_PTR,
    INT32_PTR,
    UINT16_PTR,
    ensure_cpu_float32_contiguous,
    tensor_float_ptr,
)
from .rsr_nonsquare import _V33_NCOLS_THRESHOLD

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1_58", "cpu"
)

# ---------------------------------------------------------------------------
# Lazy kernel loaders
# ---------------------------------------------------------------------------

_RSR_V33_LIB = None
_RSR_V31_LIB = None
_RSR_V33_BATCH_LIB = None
_RSR_V31_BATCH_LIB = None


def _load_rsr_v33_lib():
    global _RSR_V33_LIB
    if _RSR_V33_LIB is None:
        lib = ctypes.CDLL(os.path.join(_KERNEL_DIR, "rsr_ternary_v3_3.so"))
        lib.rsr_ternary_gemv_v3_3.restype = None
        lib.rsr_ternary_gemv_v3_3.argtypes = [
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
        _RSR_V33_LIB = lib
    return _RSR_V33_LIB


def _load_rsr_v31_lib():
    global _RSR_V31_LIB
    if _RSR_V31_LIB is None:
        lib = ctypes.CDLL(os.path.join(_KERNEL_DIR, "rsr_ternary_v3_1.so"))
        lib.rsr_ternary_gemv_v3_1.restype = None
        lib.rsr_ternary_gemv_v3_1.argtypes = [
            UINT16_PTR,
            UINT16_PTR,
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
        _RSR_V31_LIB = lib
    return _RSR_V31_LIB


def _load_rsr_v31_batch_lib():
    global _RSR_V31_BATCH_LIB
    if _RSR_V31_BATCH_LIB is None:
        lib = ctypes.CDLL(os.path.join(_KERNEL_DIR, "rsr_ternary_v3_1_batch.so"))

        # Single-layer fused act_quant + GEMV (v3.1)
        lib.rsr_ternary_gemv_v3_1_fused.restype = None
        lib.rsr_ternary_gemv_v3_1_fused.argtypes = [
            UINT16_PTR,                         # perms
            UINT16_PTR,                         # group_ends
            INT32_PTR,                          # scatter_offsets
            INT8_PTR,                           # scatter_rows
            INT8_PTR,                           # scatter_signs
            INT32_PTR,                          # block_meta
            FLOAT_PTR,                          # v_raw
            FLOAT_PTR,                          # out
            FLOAT_PTR,                          # v_scratch
            ctypes.c_int,                       # n_cols
            ctypes.c_int,                       # k
            ctypes.c_int,                       # num_blocks
        ]

        # Multi-layer batched fused act_quant + GEMV (v3.1)
        lib.rsr_ternary_gemv_v3_1_batch_fused.restype = None
        lib.rsr_ternary_gemv_v3_1_batch_fused.argtypes = [
            ctypes.c_int,                       # num_layers
            ctypes.POINTER(UINT16_PTR),         # perms_arr
            ctypes.POINTER(UINT16_PTR),         # group_ends_arr
            ctypes.POINTER(INT32_PTR),          # scatter_offsets_arr
            ctypes.POINTER(INT8_PTR),           # scatter_rows_arr
            ctypes.POINTER(INT8_PTR),           # scatter_signs_arr
            ctypes.POINTER(INT32_PTR),          # block_meta_arr
            INT32_PTR,                          # k_arr
            INT32_PTR,                          # num_blocks_arr
            FLOAT_PTR,                          # v_raw
            ctypes.POINTER(FLOAT_PTR),          # out_arr
            FLOAT_PTR,                          # v_scratch
            ctypes.c_int,                       # n_cols
        ]

        _RSR_V31_BATCH_LIB = lib
    return _RSR_V31_BATCH_LIB


def _load_rsr_v33_batch_lib():
    global _RSR_V33_BATCH_LIB
    if _RSR_V33_BATCH_LIB is None:
        lib = ctypes.CDLL(os.path.join(_KERNEL_DIR, "rsr_ternary_v3_3_batch.so"))

        # Single-layer fused act_quant + GEMV
        lib.rsr_ternary_gemv_v3_3_fused.restype = None
        lib.rsr_ternary_gemv_v3_3_fused.argtypes = [
            UINT16_PTR,                         # perms
            UINT16_PTR,                         # group_ends
            UINT16_PTR,                         # pos_masks
            UINT16_PTR,                         # neg_masks
            INT32_PTR,                          # block_meta
            FLOAT_PTR,                          # v_raw
            FLOAT_PTR,                          # out
            FLOAT_PTR,                          # v_scratch
            ctypes.c_int,                       # n_cols
            ctypes.c_int,                       # k
            ctypes.c_int,                       # num_blocks
        ]

        # Multi-layer batched fused act_quant + GEMV
        lib.rsr_ternary_gemv_v3_3_batch_fused.restype = None
        lib.rsr_ternary_gemv_v3_3_batch_fused.argtypes = [
            ctypes.c_int,                       # num_layers
            ctypes.POINTER(UINT16_PTR),         # perms_arr
            ctypes.POINTER(UINT16_PTR),         # group_ends_arr
            ctypes.POINTER(UINT16_PTR),         # pos_masks_arr
            ctypes.POINTER(UINT16_PTR),         # neg_masks_arr
            ctypes.POINTER(INT32_PTR),          # block_meta_arr
            INT32_PTR,                          # k_arr
            INT32_PTR,                          # num_blocks_arr
            FLOAT_PTR,                          # v_raw
            ctypes.POINTER(FLOAT_PTR),          # out_arr
            FLOAT_PTR,                          # v_scratch
            ctypes.c_int,                       # n_cols
        ]

        _RSR_V33_BATCH_LIB = lib
    return _RSR_V33_BATCH_LIB


# ---------------------------------------------------------------------------
# Tensor keys expected in the safetensors file for each kernel variant
# ---------------------------------------------------------------------------

# v3.3 tensors (bitmask scatter)
V33_TENSOR_KEYS = ("perms", "group_ends", "pos_masks", "neg_masks", "block_meta")
# v3.1 tensors (direct scatter)
V31_TENSOR_KEYS = (
    "perms", "group_ends", "scatter_offsets", "scatter_rows", "scatter_signs",
    "block_meta",
)


def select_cpu_tensor_keys(
    n_cols: int, k: int,
) -> tuple[str, ...]:
    """Pick v3.3 or v3.1 tensor keys based on the kernel selection threshold."""
    if n_cols >= _V33_NCOLS_THRESHOLD and k <= 16:
        return V33_TENSOR_KEYS
    return V31_TENSOR_KEYS


def uses_v33(n_cols: int, k: int) -> bool:
    """Return True if a layer with these dimensions uses the v3.3 kernel."""
    return n_cols >= _V33_NCOLS_THRESHOLD and k <= 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint16(tensor: torch.Tensor, name: str) -> np.ndarray:
    cpu = tensor.detach().to(device="cpu", dtype=torch.int32).contiguous()
    if cpu.numel() and (
        cpu.min().item() < 0 or cpu.max().item() > np.iinfo(np.uint16).max
    ):
        raise ValueError(f"Tensor {name!r} contains values outside uint16 range")
    return cpu.numpy().astype(np.uint16, copy=True)


def _to_int32(tensor: torch.Tensor, name: str) -> np.ndarray:
    cpu = tensor.detach().to(device="cpu", dtype=torch.int32).contiguous()
    if not cpu.is_contiguous():
        raise ValueError(f"Tensor {name!r} must be contiguous")
    return cpu.numpy().copy()


def _to_int8(tensor: torch.Tensor, name: str) -> np.ndarray:
    cpu = tensor.detach().to(device="cpu", dtype=torch.int8).contiguous()
    return cpu.numpy().copy()


# ---------------------------------------------------------------------------
# RSRPreprocessedMultiplier — single-layer CPU GEMV
# ---------------------------------------------------------------------------

class RSRPreprocessedMultiplier:
    """GEMV runtime built directly from saved RSR tensors.

    Automatically selects v3.3 (bitmask scatter) or v3.1 (direct scatter)
    based on n_cols, matching the threshold in rsr_nonsquare.py.
    """

    def __init__(
        self,
        layer_name: str,
        layer_meta: dict[str, Any],
        tensors: dict[str, torch.Tensor],
    ):
        self.layer_name = layer_name
        self.n_rows = int(layer_meta["n_rows"])
        self.n_cols = int(layer_meta["n_cols"])
        self.k = int(layer_meta["k"])

        self._use_v33 = uses_v33(self.n_cols, self.k)

        self._perms = _to_uint16(tensors["perms"], "perms")
        self._group_ends = _to_uint16(tensors["group_ends"], "group_ends")
        self._block_meta = _to_int32(tensors["block_meta"], "block_meta")

        if self._block_meta.size % 2 != 0:
            raise ValueError(
                f"Layer {layer_name!r} has invalid block_meta length {self._block_meta.size}"
            )

        self._num_blocks = self._block_meta.size // 2
        self._perms_ptr = self._perms.ctypes.data_as(UINT16_PTR)
        self._group_ends_ptr = self._group_ends.ctypes.data_as(UINT16_PTR)
        self._block_meta_ptr = self._block_meta.ctypes.data_as(INT32_PTR)

        if self._use_v33:
            self._pos_masks = _to_uint16(tensors["pos_masks"], "pos_masks")
            self._neg_masks = _to_uint16(tensors["neg_masks"], "neg_masks")
            self._pos_masks_ptr = self._pos_masks.ctypes.data_as(UINT16_PTR)
            self._neg_masks_ptr = self._neg_masks.ctypes.data_as(UINT16_PTR)
        else:
            self._scatter_offsets = _to_int32(
                tensors["scatter_offsets"], "scatter_offsets"
            )
            self._scatter_rows = _to_int8(
                tensors["scatter_rows"], "scatter_rows"
            )
            self._scatter_signs = _to_int8(
                tensors["scatter_signs"], "scatter_signs"
            )
            self._scatter_offsets_ptr = self._scatter_offsets.ctypes.data_as(INT32_PTR)
            self._scatter_rows_ptr = self._scatter_rows.ctypes.data_as(INT8_PTR)
            self._scatter_signs_ptr = self._scatter_signs.ctypes.data_as(INT8_PTR)

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 1 or vector.shape[0] != self.n_cols:
            raise ValueError(
                f"Layer {self.layer_name!r} expected vector of shape ({self.n_cols},), "
                f"got {tuple(vector.shape)}"
            )

        v_cpu = ensure_cpu_float32_contiguous(vector)
        out_cpu = torch.empty(self.n_rows, dtype=torch.float32)

        if self._use_v33:
            _load_rsr_v33_lib().rsr_ternary_gemv_v3_3(
                self._perms_ptr,
                self._group_ends_ptr,
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
            _load_rsr_v31_lib().rsr_ternary_gemv_v3_1(
                self._perms_ptr,
                self._group_ends_ptr,
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
        return out_cpu

    def fused_call(self, v_raw: torch.Tensor, v_scratch: np.ndarray) -> torch.Tensor:
        """act_quant + GEMV in a single C call (v3.3 and v3.1)."""
        v_cpu = ensure_cpu_float32_contiguous(v_raw)
        out_cpu = torch.empty(self.n_rows, dtype=torch.float32)
        if self._use_v33:
            _load_rsr_v33_batch_lib().rsr_ternary_gemv_v3_3_fused(
                self._perms_ptr,
                self._group_ends_ptr,
                self._pos_masks_ptr,
                self._neg_masks_ptr,
                self._block_meta_ptr,
                tensor_float_ptr(v_cpu),
                tensor_float_ptr(out_cpu),
                ctypes.cast(v_scratch.ctypes.data, FLOAT_PTR),
                self.n_cols,
                self.k,
                self._num_blocks,
            )
        else:
            _load_rsr_v31_batch_lib().rsr_ternary_gemv_v3_1_fused(
                self._perms_ptr,
                self._group_ends_ptr,
                self._scatter_offsets_ptr,
                self._scatter_rows_ptr,
                self._scatter_signs_ptr,
                self._block_meta_ptr,
                tensor_float_ptr(v_cpu),
                tensor_float_ptr(out_cpu),
                ctypes.cast(v_scratch.ctypes.data, FLOAT_PTR),
                self.n_cols,
                self.k,
                self._num_blocks,
            )
        return out_cpu


# ---------------------------------------------------------------------------
# RSRBatchMultiplier — multi-layer batched GEMV
# ---------------------------------------------------------------------------

class RSRBatchMultiplier:
    """Batched GEMV for multiple v3.3 layers sharing the same input.

    act_quant is applied once in C; all layers' GEMVs execute in a single
    ctypes call with OpenMP parallelism across the combined block pool.
    """

    def __init__(self, multipliers: list[RSRPreprocessedMultiplier]):
        self._mults = multipliers
        self.num_layers = len(multipliers)
        self.n_cols = multipliers[0].n_cols
        for m in multipliers:
            if m.n_cols != self.n_cols:
                raise ValueError(
                    "All layers in a batch must share n_cols; "
                    f"got {m.n_cols} vs {self.n_cols}"
                )

        n = self.num_layers
        self._perms_arr = (UINT16_PTR * n)(
            *(m._perms_ptr for m in multipliers)
        )
        self._ge_arr = (UINT16_PTR * n)(
            *(m._group_ends_ptr for m in multipliers)
        )
        self._pm_arr = (UINT16_PTR * n)(
            *(m._pos_masks_ptr for m in multipliers)
        )
        self._nm_arr = (UINT16_PTR * n)(
            *(m._neg_masks_ptr for m in multipliers)
        )
        self._bm_arr = (INT32_PTR * n)(
            *(m._block_meta_ptr for m in multipliers)
        )
        self._k_arr = (ctypes.c_int32 * n)(*(m.k for m in multipliers))
        self._nb_arr = (ctypes.c_int32 * n)(
            *(m._num_blocks for m in multipliers)
        )

        # Pre-allocate reusable output buffers (one per layer)
        self._out_bufs: list[torch.Tensor] = []
        self._out_ptr_arr = (FLOAT_PTR * n)()
        for i, m in enumerate(multipliers):
            buf = torch.empty(m.n_rows, dtype=torch.float32)
            self._out_bufs.append(buf)
            self._out_ptr_arr[i] = tensor_float_ptr(buf)

        # Scratch buffer for the quantised input vector
        self._v_scratch = np.empty(self.n_cols, dtype=np.float32)
        self._v_scratch_ptr = ctypes.cast(
            self._v_scratch.ctypes.data, FLOAT_PTR
        )

    def __call__(self, v_raw: torch.Tensor) -> list[torch.Tensor]:
        """Run act_quant + all GEMVs, return a list of output tensors."""
        v_cpu = ensure_cpu_float32_contiguous(v_raw)
        _load_rsr_v33_batch_lib().rsr_ternary_gemv_v3_3_batch_fused(
            self.num_layers,
            self._perms_arr,
            self._ge_arr,
            self._pm_arr,
            self._nm_arr,
            self._bm_arr,
            self._k_arr,
            self._nb_arr,
            tensor_float_ptr(v_cpu),
            self._out_ptr_arr,
            self._v_scratch_ptr,
            self.n_cols,
        )
        # Clone because the internal buffers are reused on the next call.
        return [buf.clone() for buf in self._out_bufs]


class RSRBatchMultiplierV31:
    """Batched GEMV for multiple v3.1 layers sharing the same input.

    act_quant is applied once in C; all layers' GEMVs execute in a single
    ctypes call with OpenMP parallelism across the combined block pool.
    """

    def __init__(self, multipliers: list[RSRPreprocessedMultiplier]):
        self._mults = multipliers
        self.num_layers = len(multipliers)
        self.n_cols = multipliers[0].n_cols
        for m in multipliers:
            if m.n_cols != self.n_cols:
                raise ValueError(
                    "All layers in a batch must share n_cols; "
                    f"got {m.n_cols} vs {self.n_cols}"
                )

        n = self.num_layers
        self._perms_arr = (UINT16_PTR * n)(
            *(m._perms_ptr for m in multipliers)
        )
        self._ge_arr = (UINT16_PTR * n)(
            *(m._group_ends_ptr for m in multipliers)
        )
        self._so_arr = (INT32_PTR * n)(
            *(m._scatter_offsets_ptr for m in multipliers)
        )
        self._sr_arr = (INT8_PTR * n)(
            *(m._scatter_rows_ptr for m in multipliers)
        )
        self._ss_arr = (INT8_PTR * n)(
            *(m._scatter_signs_ptr for m in multipliers)
        )
        self._bm_arr = (INT32_PTR * n)(
            *(m._block_meta_ptr for m in multipliers)
        )
        self._k_arr = (ctypes.c_int32 * n)(*(m.k for m in multipliers))
        self._nb_arr = (ctypes.c_int32 * n)(
            *(m._num_blocks for m in multipliers)
        )

        # Pre-allocate reusable output buffers (one per layer)
        self._out_bufs: list[torch.Tensor] = []
        self._out_ptr_arr = (FLOAT_PTR * n)()
        for i, m in enumerate(multipliers):
            buf = torch.empty(m.n_rows, dtype=torch.float32)
            self._out_bufs.append(buf)
            self._out_ptr_arr[i] = tensor_float_ptr(buf)

        # Scratch buffer for the quantised input vector
        self._v_scratch = np.empty(self.n_cols, dtype=np.float32)
        self._v_scratch_ptr = ctypes.cast(
            self._v_scratch.ctypes.data, FLOAT_PTR
        )

    def __call__(self, v_raw: torch.Tensor) -> list[torch.Tensor]:
        """Run act_quant + all GEMVs, return a list of output tensors."""
        v_cpu = ensure_cpu_float32_contiguous(v_raw)
        _load_rsr_v31_batch_lib().rsr_ternary_gemv_v3_1_batch_fused(
            self.num_layers,
            self._perms_arr,
            self._ge_arr,
            self._so_arr,
            self._sr_arr,
            self._ss_arr,
            self._bm_arr,
            self._k_arr,
            self._nb_arr,
            tensor_float_ptr(v_cpu),
            self._out_ptr_arr,
            self._v_scratch_ptr,
            self.n_cols,
        )
        # Clone because the internal buffers are reused on the next call.
        return [buf.clone() for buf in self._out_bufs]


# ---------------------------------------------------------------------------
# RSRBatchContext — orchestrates batch execution
# ---------------------------------------------------------------------------

class RSRBatchContext:
    """Manages a group of RSR layers that share the same input.

    The first layer to call :meth:`get_output` in a forward pass triggers
    the batch kernel for *all* layers in the group.  Subsequent layers
    retrieve their cached result.  After every layer has been served the
    cache auto-invalidates, ready for the next forward pass.
    """

    def __init__(self, batch_mult: RSRBatchMultiplier,
                 layer_names: list[str]):
        self._batch_mult = batch_mult
        self._name_to_idx = {name: i for i, name in enumerate(layer_names)}
        self._num_layers = len(layer_names)
        self._cached_outputs: list[torch.Tensor] | None = None
        self._call_count = 0

    def get_output(self, layer_name: str, v_raw: torch.Tensor) -> torch.Tensor:
        if self._call_count == 0:
            self._cached_outputs = self._batch_mult(v_raw)
        self._call_count += 1
        out = self._cached_outputs[self._name_to_idx[layer_name]]
        if self._call_count >= self._num_layers:
            self._cached_outputs = None
            self._call_count = 0
        return out
