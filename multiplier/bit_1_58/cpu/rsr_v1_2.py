"""
Ternary RSR v1.2 — single ternary column encoding.

Instead of two separate binary encodings, encode each column as a 2k-bit
integer: upper k bits encode M_pos pattern, lower k bits encode M_neg pattern.
This gives a single permutation per block, a single aggregation, and a single
signed scatter pass.

Trade-off: up to 3^k unique patterns per block (vs 2^k for binary), but only
one permutation + aggregation instead of two.
"""

import torch

from multiplier.base import Multiplier


class RSRTernaryV1_2Multiplier(Multiplier):
    """Ternary RSR v1.2: unified 2k-bit encoding, single pass."""

    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        assert self.n % k == 0, f"n={self.n} must be divisible by k={k}"
        assert 2 * k <= 62, f"2*k={2*k} exceeds int64 bit width"
        super().__init__(M)
        self.prep()

    def prep(self):
        n, k = self.n, self.k
        device = self.M.device
        num_blocks = n // k

        M_pos = (self.M == 1).long()
        M_neg = (self.M == -1).long()

        # Bit weights for upper (pos) and lower (neg) k bits
        bit_weights_pos = (1 << torch.arange(k - 1, -1, -1, dtype=torch.int64)) << k  # shifted left by k
        bit_weights_neg = (1 << torch.arange(k - 1, -1, -1, dtype=torch.int64))

        perms = []
        group_indices = []
        scatter_sign_masks = []  # (num_unique, k) with values +1, -1, 0

        for b in range(num_blocks):
            block_pos = M_pos[b * k : (b + 1) * k, :]  # (k, n)
            block_neg = M_neg[b * k : (b + 1) * k, :]  # (k, n)

            # Encode: upper k bits = pos pattern, lower k bits = neg pattern
            col_values = (bit_weights_pos @ block_pos.cpu()).to(device) + \
                         (bit_weights_neg @ block_neg.cpu()).to(device)  # (n,)

            perm = torch.argsort(col_values, stable=True)
            perms.append(perm)

            sorted_values = col_values[perm]
            uniq, inverse = torch.unique(sorted_values, return_inverse=True)
            group_indices.append(inverse)

            # Decode unique patterns to signed scatter masks
            shifts_pos = torch.arange(k - 1, -1, -1, device=device) + k  # upper bits
            shifts_neg = torch.arange(k - 1, -1, -1, device=device)     # lower bits
            pos_bits = ((uniq.unsqueeze(1) >> shifts_pos) & 1).float()  # (num_unique, k)
            neg_bits = ((uniq.unsqueeze(1) >> shifts_neg) & 1).float()  # (num_unique, k)
            sign_mask = pos_bits - neg_bits  # +1 where pos, -1 where neg, 0 elsewhere
            scatter_sign_masks.append(sign_mask)

        self._perms = torch.stack(perms)  # (num_blocks, n)
        self._group_indices = group_indices
        self._scatter_sign_masks = scatter_sign_masks  # list of (num_unique, k)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        n, k = self.n, self.k
        num_blocks = n // k

        v_perm = v[self._perms]  # (num_blocks, n)

        results = []
        for b in range(num_blocks):
            num_unique = self._scatter_sign_masks[b].shape[0]
            aggregated = torch.zeros(num_unique, dtype=v.dtype, device=v.device)
            aggregated.scatter_add_(0, self._group_indices[b], v_perm[b])

            # Signed scatter: (k, num_unique) @ (num_unique,) → (k,)
            result = self._scatter_sign_masks[b].t() @ aggregated
            results.append(result)

        return torch.cat(results)
