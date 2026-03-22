"""Shared preprocessing for ternary CUDA RSR multipliers."""

import ctypes
import os

import numpy as np
import torch

_PREP_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "kernels",
    "bit_1_58",
    "cpu",
)

_prep_lib = ctypes.CDLL(os.path.join(_PREP_DIR, "rsr_ternary_prep.so"))
_prep_lib.rsr_ternary_prep.restype = None
_prep_lib.rsr_ternary_prep.argtypes = [
    ctypes.POINTER(ctypes.c_int8),
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


def _build_group_starts(group_ends: np.ndarray, block_meta: np.ndarray, num_blocks: int) -> np.ndarray:
    starts = np.empty_like(group_ends)
    for b in range(num_blocks):
        g_off = int(block_meta[b * 2])
        n_groups = int(block_meta[b * 2 + 1])
        for g in range(n_groups):
            gg = g_off + g
            starts[gg] = 0 if g == 0 else group_ends[gg - 1]
    return starts


def _build_cuda_block_meta(raw_block_meta: np.ndarray, total_groups: int, num_blocks: int) -> np.ndarray:
    out = np.empty(2 * num_blocks, dtype=np.int32)
    for b in range(num_blocks):
        g_off = int(raw_block_meta[b * 2])
        next_g_off = total_groups if b + 1 == num_blocks else int(raw_block_meta[(b + 1) * 2])
        out[b * 2] = g_off
        out[b * 2 + 1] = next_g_off - g_off
    return out


def sort_perms_within_groups(
    perms: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    block_meta: np.ndarray,
    n: int,
    num_blocks: int,
) -> np.ndarray:
    sorted_perms = perms.copy()
    for b in range(num_blocks):
        g_off = int(block_meta[b * 2])
        n_groups = int(block_meta[b * 2 + 1])
        base = b * n
        for g in range(n_groups):
            gg = g_off + g
            start = int(group_starts[gg])
            end = int(group_ends[gg])
            sorted_perms[base + start : base + end].sort()
    return sorted_perms


def build_group_sign_masks(
    scatter_offsets: np.ndarray,
    scatter_rows: np.ndarray,
    scatter_signs: np.ndarray,
    total_groups: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert k <= 16, f"sign-mask path requires k <= 16, got k={k}"
    pos_masks = np.zeros(total_groups, dtype=np.int32)
    neg_masks = np.zeros(total_groups, dtype=np.int32)
    for g in range(total_groups):
        pos_mask = 0
        neg_mask = 0
        for s in range(int(scatter_offsets[g]), int(scatter_offsets[g + 1])):
            bit = 1 << int(scatter_rows[s])
            if int(scatter_signs[s]) > 0:
                pos_mask |= bit
            else:
                neg_mask |= bit
        pos_masks[g] = pos_mask
        neg_masks[g] = neg_mask
    return pos_masks, neg_masks


def pack_group_metadata(
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    pos_masks: np.ndarray,
    neg_masks: np.ndarray,
) -> np.ndarray:
    return np.stack([group_starts, group_ends, pos_masks, neg_masks], axis=1).astype(np.int32, copy=False)


def pack_group_metadata_u16(
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    pos_masks: np.ndarray,
    neg_masks: np.ndarray,
) -> np.ndarray:
    lengths = group_ends - group_starts
    return np.stack(
        [
            group_starts.astype(np.uint16, copy=False),
            lengths.astype(np.uint16, copy=False),
            pos_masks.astype(np.uint16, copy=False),
            neg_masks.astype(np.uint16, copy=False),
        ],
        axis=1,
    ).astype(np.uint16, copy=False)


def prep_ternary_on_cpu(
    M: torch.Tensor,
    n: int,
    k: int,
) -> dict:
    num_blocks = n // k
    num_buckets = 1 << (2 * k)

    M_np = M.detach().cpu().to(torch.int8).numpy().copy()
    max_groups = num_blocks * min(num_buckets, n)
    max_scatter = max_groups * 2 * k

    perms = np.empty(num_blocks * n, dtype=np.int32)
    group_ends = np.empty(max_groups, dtype=np.int32)
    scatter_offsets = np.empty(max_groups + 1, dtype=np.int32)
    scatter_rows = np.empty(max_scatter, dtype=np.int8)
    scatter_signs = np.empty(max_scatter, dtype=np.int8)
    raw_block_meta = np.empty(2 * num_blocks, dtype=np.int32)
    out_sizes = np.empty(2, dtype=np.int32)

    _prep_lib.rsr_ternary_prep(
        M_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        n,
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


def prep_ternary_on_cpu_move_to_cuda(M: torch.Tensor, n: int, k: int, device: torch.device) -> dict:
    data = prep_ternary_on_cpu(M, n, k)
    return {key: (torch.from_numpy(value.copy()).to(device) if isinstance(value, np.ndarray) else value) for key, value in data.items()}
