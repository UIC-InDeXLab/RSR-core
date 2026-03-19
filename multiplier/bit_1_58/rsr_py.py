"""
Ternary RSR v1.0 — decompose M into M_1 - M_2 (both binary),
run binary RSR on each, subtract results.

This is the baseline ternary RSR using pure PyTorch.
"""

import torch

from multiplier.base import Multiplier


def _binary_rsr_prep(M_bin: torch.Tensor, k: int):
    """Preprocess a binary matrix for RSR inference.

    Returns per-block: perms, group_indices, unique_bits.
    """
    n = M_bin.shape[0]
    num_blocks = n // k
    device = M_bin.device

    perms = []
    group_indices = []
    unique_bits = []

    for b in range(num_blocks):
        block = M_bin[b * k : (b + 1) * k, :]  # (k, n)

        bit_weights = (1 << torch.arange(k - 1, -1, -1, dtype=torch.int64))
        col_values = (bit_weights @ block.cpu().long()).to(device)  # (n,)

        perm = torch.argsort(col_values, stable=True)
        perms.append(perm)

        sorted_values = col_values[perm]
        uniq, inverse = torch.unique(sorted_values, return_inverse=True)
        group_indices.append(inverse)

        shifts = torch.arange(k - 1, -1, -1, device=device)
        bits = ((uniq.unsqueeze(1) >> shifts) & 1).float()  # (num_unique, k)
        unique_bits.append(bits)

    return torch.stack(perms), group_indices, unique_bits


def _binary_rsr_infer(v, perms, group_indices, unique_bits, n, k):
    """Run RSR inference for a preprocessed binary matrix."""
    num_blocks = n // k

    v_perm = v[perms]  # (num_blocks, n)

    results = []
    for b in range(num_blocks):
        num_unique = unique_bits[b].shape[0]
        aggregated = torch.zeros(num_unique, dtype=v.dtype, device=v.device)
        aggregated.scatter_add_(0, group_indices[b], v_perm[b])
        result = unique_bits[b].t() @ aggregated  # (k,)
        results.append(result)

    return torch.cat(results)


class RSRTernaryV1_0Multiplier(Multiplier):
    """Ternary RSR v1.0: two independent binary RSR instances.

    M = M_1 - M_2 where M_1 = (M == 1), M_2 = (M == -1).
    Result = RSR(M_1, v) - RSR(M_2, v).
    """

    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        assert self.n % k == 0, f"n={self.n} must be divisible by k={k}"
        super().__init__(M)
        self.prep()

    def prep(self):
        M_pos = (self.M == 1).float()   # M_1: where M is +1
        M_neg = (self.M == -1).float()  # M_2: where M is -1

        self._pos_perms, self._pos_groups, self._pos_bits = _binary_rsr_prep(M_pos, self.k)
        self._neg_perms, self._neg_groups, self._neg_bits = _binary_rsr_prep(M_neg, self.k)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        y_pos = _binary_rsr_infer(v, self._pos_perms, self._pos_groups, self._pos_bits, self.n, self.k)
        y_neg = _binary_rsr_infer(v, self._neg_perms, self._neg_groups, self._neg_bits, self.n, self.k)
        return y_pos - y_neg
