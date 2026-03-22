"""
Shared preprocessing for non-square CUDA RSR multipliers.

Runs the C counting-sort prep kernel (non-square variant) on CPU,
then moves results to GPU tensors.
"""

import ctypes
import os
import numpy as np
import torch

_prep_lib_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cpu", "rsr_prep_nonsquare.so"
)
_prep_lib = ctypes.CDLL(_prep_lib_path)
_prep_lib.rsr_prep_nonsquare_omp.restype = None
_prep_lib.rsr_prep_nonsquare_omp.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # M
    ctypes.c_int,                      # n_rows
    ctypes.c_int,                      # n_cols
    ctypes.c_int,                      # k
    ctypes.POINTER(ctypes.c_int32),   # perms
    ctypes.POINTER(ctypes.c_int32),   # group_ends
    ctypes.POINTER(ctypes.c_int32),   # scatter_offsets
    ctypes.POINTER(ctypes.c_uint8),   # scatter_rows
    ctypes.POINTER(ctypes.c_int32),   # block_meta
    ctypes.POINTER(ctypes.c_int32),   # out_sizes
]


def prep_nonsquare_on_cpu_move_to_cuda(
    M: torch.Tensor, n_rows: int, n_cols: int, k: int, device: torch.device
) -> dict:
    """Run non-square CPU preprocessing and return GPU tensors."""
    num_blocks = n_rows // k
    num_buckets = 1 << k

    M_np = M.cpu().numpy().astype(np.uint8).copy()

    max_groups = num_blocks * min(num_buckets, n_cols)
    max_scatter = max_groups * k

    perms = np.empty(num_blocks * n_cols, dtype=np.int32)
    group_ends = np.empty(max_groups, dtype=np.int32)
    scatter_offsets = np.empty(max_groups + 1, dtype=np.int32)
    scatter_rows = np.empty(max_scatter, dtype=np.uint8)
    block_meta = np.empty(2 * num_blocks, dtype=np.int32)
    out_sizes = np.empty(2, dtype=np.int32)

    _prep_lib.rsr_prep_nonsquare_omp(
        M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        n_rows, n_cols, k,
        perms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        group_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        scatter_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        scatter_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        block_meta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    total_groups = int(out_sizes[0])
    total_scatter = int(out_sizes[1])

    return {
        "perms": torch.from_numpy(perms.copy()).to(device),
        "group_ends": torch.from_numpy(group_ends[:total_groups].copy()).to(device),
        "scatter_offsets": torch.from_numpy(scatter_offsets[:total_groups + 1].copy()).to(device),
        "scatter_rows": torch.from_numpy(scatter_rows[:total_scatter].copy()).to(device),
        "block_meta": torch.from_numpy(block_meta.copy()).to(device),
        "num_blocks": num_blocks,
    }
