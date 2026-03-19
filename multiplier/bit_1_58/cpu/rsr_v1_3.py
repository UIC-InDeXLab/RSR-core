"""
Ternary RSR v1.3 — padded batch processing across all blocks.

Builds on v1.2's single ternary encoding but pads all blocks to the same
num_unique count, enabling fully batched scatter_add and matmul across blocks.
Eliminates the Python for-loop over blocks during inference.
"""

import torch

from multiplier.base import Multiplier


class RSRTernaryV1_3Multiplier(Multiplier):
    """Ternary RSR v1.3: batched inference over all blocks."""

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

        bit_weights_pos = (1 << torch.arange(k - 1, -1, -1, dtype=torch.int64)) << k
        bit_weights_neg = (1 << torch.arange(k - 1, -1, -1, dtype=torch.int64))

        perms_list = []
        group_indices_list = []
        sign_masks_list = []

        for b in range(num_blocks):
            block_pos = M_pos[b * k : (b + 1) * k, :]
            block_neg = M_neg[b * k : (b + 1) * k, :]

            col_values = (bit_weights_pos @ block_pos.cpu()).to(device) + \
                         (bit_weights_neg @ block_neg.cpu()).to(device)

            perm = torch.argsort(col_values, stable=True)
            perms_list.append(perm)

            sorted_values = col_values[perm]
            uniq, inverse = torch.unique(sorted_values, return_inverse=True)
            group_indices_list.append(inverse)

            shifts_pos = torch.arange(k - 1, -1, -1, device=device) + k
            shifts_neg = torch.arange(k - 1, -1, -1, device=device)
            pos_bits = ((uniq.unsqueeze(1) >> shifts_pos) & 1).float()
            neg_bits = ((uniq.unsqueeze(1) >> shifts_neg) & 1).float()
            sign_masks_list.append(pos_bits - neg_bits)  # (num_unique_b, k)

        self._perms = torch.stack(perms_list)  # (num_blocks, n)

        # Pad group_indices and sign_masks to max_unique for batched ops
        max_unique = max(sm.shape[0] for sm in sign_masks_list)
        self._max_unique = max_unique

        # Padded group indices: (num_blocks, n), values in [0, max_unique)
        # Entries mapping to padded slots aggregate into a dummy bucket (ignored)
        group_indices_padded = torch.stack(group_indices_list)  # (num_blocks, n)
        self._group_indices = group_indices_padded

        # Padded sign masks: (num_blocks, max_unique, k), padded rows are zero
        sign_masks_padded = torch.zeros(num_blocks, max_unique, k, dtype=torch.float32, device=device)
        for b in range(num_blocks):
            nu = sign_masks_list[b].shape[0]
            sign_masks_padded[b, :nu, :] = sign_masks_list[b]
        self._sign_masks = sign_masks_padded  # (num_blocks, max_unique, k)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        n, k = self.n, self.k
        num_blocks = n // k

        # Permute: (num_blocks, n)
        v_perm = v[self._perms]

        # Aggregate: batched scatter_add into (num_blocks, max_unique)
        aggregated = torch.zeros(num_blocks, self._max_unique, dtype=v.dtype, device=v.device)
        aggregated.scatter_add_(1, self._group_indices, v_perm)

        # Signed scatter: batched matmul (num_blocks, k, max_unique) @ (num_blocks, max_unique, 1)
        # = (num_blocks, k, 1) → squeeze to (num_blocks, k)
        result = torch.bmm(
            self._sign_masks.transpose(1, 2),  # (num_blocks, k, max_unique)
            aggregated.unsqueeze(2),            # (num_blocks, max_unique, 1)
        ).squeeze(2)  # (num_blocks, k)

        return result.reshape(n)
