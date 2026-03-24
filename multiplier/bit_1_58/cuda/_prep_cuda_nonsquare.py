"""Shared preprocessing for non-square ternary CUDA RSR multipliers.

Calls the C preprocessing kernel (rsr_ternary_prep_nonsquare) on CPU,
then moves all metadata tensors to CUDA.
"""

import ctypes
import os

import numpy as np
import torch

from ._prep_cuda import (
    _build_cuda_block_meta,
    _build_group_starts,
    build_group_sign_masks,
    pack_group_metadata,
    sort_perms_within_groups,
)

_PREP_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "kernels",
    "bit_1_58",
    "cpu",
)

_prep_lib = ctypes.CDLL(os.path.join(_PREP_DIR, "rsr_ternary_prep_nonsquare.so"))
_prep_lib.rsr_ternary_prep_nonsquare.restype = None
_prep_lib.rsr_ternary_prep_nonsquare.argtypes = [
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]


def prep_ternary_nonsquare_on_cpu(
    M: torch.Tensor,
    n_rows: int,
    n_cols: int,
    k: int,
) -> dict:
    num_blocks = n_rows // k
    num_buckets = 1 << (2 * k)

    M_np = M.detach().cpu().to(torch.int8).numpy().copy()
    max_groups = num_blocks * min(num_buckets, n_cols)
    max_scatter = max_groups * 2 * k

    perms = np.empty(num_blocks * n_cols, dtype=np.int32)
    group_ends = np.empty(max_groups, dtype=np.int32)
    scatter_offsets = np.empty(max_groups + 1, dtype=np.int32)
    scatter_rows = np.empty(max_scatter, dtype=np.int8)
    scatter_signs = np.empty(max_scatter, dtype=np.int8)
    raw_block_meta = np.empty(2 * num_blocks, dtype=np.int32)
    out_sizes = np.empty(2, dtype=np.int32)

    _prep_lib.rsr_ternary_prep_nonsquare(
        M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        n_rows,
        n_cols,
        k,
        perms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        group_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        scatter_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        scatter_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scatter_signs.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        raw_block_meta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    total_groups = int(out_sizes[0])
    total_scatter = int(out_sizes[1])
    group_ends = group_ends[:total_groups].copy()
    scatter_offsets = scatter_offsets[: total_groups + 1].copy()
    scatter_rows = scatter_rows[:total_scatter].copy()
    scatter_signs = scatter_signs[:total_scatter].copy()
    block_meta = _build_cuda_block_meta(raw_block_meta, total_groups, num_blocks)
    group_starts = _build_group_starts(group_ends, block_meta, num_blocks)

    return {
        "perms": perms.copy(),
        "group_starts": group_starts,
        "group_ends": group_ends,
        "scatter_offsets": scatter_offsets,
        "scatter_rows": scatter_rows,
        "scatter_signs": scatter_signs,
        "block_meta": block_meta,
        "num_blocks": num_blocks,
        "total_groups": total_groups,
    }


def prep_ternary_nonsquare_on_cpu_move_to_cuda(
    M: torch.Tensor,
    n_rows: int,
    n_cols: int,
    k: int,
    device: torch.device,
) -> dict:
    data = prep_ternary_nonsquare_on_cpu(M, n_rows, n_cols, k)
    return {
        key: (torch.from_numpy(value.copy()).to(device) if isinstance(value, np.ndarray) else value)
        for key, value in data.items()
    }
