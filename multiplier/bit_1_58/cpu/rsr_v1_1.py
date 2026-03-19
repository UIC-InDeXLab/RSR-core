"""
Ternary RSR v1.1 — fused loop over blocks.

Instead of two full passes (one for M_1, one for M_2), we iterate blocks once
and process both pos/neg within each block iteration. This reduces Python loop
overhead and improves cache locality since v is accessed once per block.
"""

import torch

from multiplier.base import Multiplier
from multiplier.bit_1_58.rsr_py import _binary_rsr_prep


class RSRTernaryV1_1Multiplier(Multiplier):
    """Ternary RSR v1.1: fused block loop for M_1 and M_2."""

    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        assert self.n % k == 0, f"n={self.n} must be divisible by k={k}"
        super().__init__(M)
        self.prep()

    def prep(self):
        M_pos = (self.M == 1).float()
        M_neg = (self.M == -1).float()

        self._pos_perms, self._pos_groups, self._pos_bits = _binary_rsr_prep(M_pos, self.k)
        self._neg_perms, self._neg_groups, self._neg_bits = _binary_rsr_prep(M_neg, self.k)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        n, k = self.n, self.k
        num_blocks = n // k

        v_perm_pos = v[self._pos_perms]  # (num_blocks, n)
        v_perm_neg = v[self._neg_perms]  # (num_blocks, n)

        results = []
        for b in range(num_blocks):
            # Positive part
            num_unique_pos = self._pos_bits[b].shape[0]
            agg_pos = torch.zeros(num_unique_pos, dtype=v.dtype, device=v.device)
            agg_pos.scatter_add_(0, self._pos_groups[b], v_perm_pos[b])
            res_pos = self._pos_bits[b].t() @ agg_pos

            # Negative part
            num_unique_neg = self._neg_bits[b].shape[0]
            agg_neg = torch.zeros(num_unique_neg, dtype=v.dtype, device=v.device)
            agg_neg.scatter_add_(0, self._neg_groups[b], v_perm_neg[b])
            res_neg = self._neg_bits[b].t() @ agg_neg

            results.append(res_pos - res_neg)

        return torch.cat(results)
