"""
Shared preprocessing for CUDA RSR multipliers.

Runs the C counting-sort prep kernel on CPU, then moves results to GPU tensors.
"""

import ctypes
import os
import numpy as np
import torch

_prep_lib_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1", "cpu", "rsr_prep.so"
)
_prep_lib = ctypes.CDLL(_prep_lib_path)
_prep_lib.rsr_prep_omp.restype = None
_prep_lib.rsr_prep_omp.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]


def prep_on_cpu_move_to_cuda(M: torch.Tensor, n: int, k: int, device: torch.device) -> dict:
    """Run CPU preprocessing and return GPU tensors."""
    num_blocks = n // k
    num_buckets = 1 << k

    M_np = M.cpu().numpy().astype(np.uint8).copy()

    max_groups = num_blocks * min(num_buckets, n)
    max_scatter = max_groups * k

    perms = np.empty(num_blocks * n, dtype=np.int32)
    group_ends = np.empty(max_groups, dtype=np.int32)
    scatter_offsets = np.empty(max_groups + 1, dtype=np.int32)
    scatter_rows = np.empty(max_scatter, dtype=np.uint8)
    block_meta = np.empty(2 * num_blocks, dtype=np.int32)
    out_sizes = np.empty(2, dtype=np.int32)

    _prep_lib.rsr_prep_omp(
        M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        n, k,
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
