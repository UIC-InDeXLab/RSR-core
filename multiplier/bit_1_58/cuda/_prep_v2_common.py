"""Shared preprocessing for ternary CUDA RSR v2.x multipliers.

This path keeps the algorithm from ``ALGORITHM.md`` intact while making the
runtime metadata friendlier for CUDA kernels:
  - zero-pattern groups are omitted because they never scatter to any row
  - each remaining group is packed into one 64-bit word:
      [ start(u16) | length(u16) | pos_mask(u16) | neg_mask(u16) ]
"""

from __future__ import annotations

import numpy as np
import torch

from ._prep_cuda import build_group_sign_masks
from ._prep_cuda_nonsquare import prep_ternary_nonsquare_on_cpu


def prep_compact_u64(
    M: torch.Tensor,
    n_rows_orig: int,
    n_cols: int,
    k: int,
    n_rows_padded: int,
    device: torch.device,
):
    """Build compact metadata for v2.x kernels.

    The preprocessing steps still follow the ternary RSR algorithm:
      1. encode columns and group identical ternary codes
      2. aggregate each group over permuted input indices
      3. scatter each aggregate using signed row masks

    We only drop groups whose positive and negative masks are both zero, since
    those correspond to an all-zero ternary pattern and contribute nothing to
    the output slice for that block.
    """
    if not (0 < k <= 16):
        raise ValueError(f"v2 metadata path requires 0 < k <= 16, got k={k}")
    if n_cols > 65535:
        raise ValueError(f"v2 metadata path requires n_cols <= 65535, got n_cols={n_cols}")

    if n_rows_padded > n_rows_orig:
        M_padded = torch.zeros(n_rows_padded, n_cols, dtype=M.dtype, device=M.device)
        M_padded[:n_rows_orig, :] = M
    else:
        M_padded = M

    data = prep_ternary_nonsquare_on_cpu(M_padded, n_rows_padded, n_cols, k)
    num_blocks = data["num_blocks"]
    perms_np = data["perms"].copy()
    group_starts = data["group_starts"]
    group_ends = data["group_ends"]
    block_meta = data["block_meta"]

    pos_masks, neg_masks = build_group_sign_masks(
        data["scatter_offsets"],
        data["scatter_rows"],
        data["scatter_signs"],
        data["total_groups"],
        k,
    )

    packed_groups = []
    compact_block_meta = np.empty(2 * num_blocks, dtype=np.int32)

    for b in range(num_blocks):
        g_off = int(block_meta[b * 2])
        n_groups = int(block_meta[b * 2 + 1])
        base = b * n_cols
        kept = 0
        compact_block_meta[b * 2] = len(packed_groups)

        for g in range(n_groups):
            gg = g_off + g
            start = int(group_starts[gg])
            end = int(group_ends[gg])
            pos_mask = int(pos_masks[gg])
            neg_mask = int(neg_masks[gg])

            if (pos_mask | neg_mask) == 0:
                continue

            perms_np[base + start : base + end].sort()
            length = end - start
            packed = (
                np.uint64(start)
                | (np.uint64(length) << np.uint64(16))
                | (np.uint64(pos_mask & 0xFFFF) << np.uint64(32))
                | (np.uint64(neg_mask & 0xFFFF) << np.uint64(48))
            )
            packed_groups.append(packed)
            kept += 1

        compact_block_meta[b * 2 + 1] = kept

    perms_u16 = torch.from_numpy(perms_np.astype(np.uint16, copy=False)).to(device)

    if packed_groups:
        group_words = np.asarray(packed_groups, dtype=np.uint64)
        group_packed_u64 = torch.from_numpy(group_words.view(np.int64).copy()).to(device)
    else:
        group_packed_u64 = torch.empty(0, dtype=torch.int64, device=device)

    block_meta_gpu = torch.from_numpy(compact_block_meta).int().to(device)
    return perms_u16, group_packed_u64, block_meta_gpu, num_blocks
