import torch

from .base import Multiplier


class RSRPythonMultiplier(Multiplier):
    def __init__(self, M: torch.Tensor, k: int):
        self.n = M.shape[0]
        self.k = k
        assert self.n % k == 0, f"n={self.n} must be divisible by k={k}"
        super().__init__(M)
        self.prep()

    def prep(self):
        n, k = self.n, self.k
        device = self.M.device
        num_blocks = n // k

        perms = []
        # For each block: maps each column to its index in the unique pattern list
        group_indices = []
        # For each block: (num_unique, k) binary matrix of unique patterns
        unique_bits = []

        for b in range(num_blocks):
            block = self.M[b * k : (b + 1) * k, :]  # (k, n)

            # Convert each column to an integer: row 0 is MSB
            # Use CPU for int64 matmul (not supported on CUDA), then move back
            bit_weights = (1 << torch.arange(k - 1, -1, -1, dtype=torch.int64))
            col_values = (bit_weights @ block.cpu().long()).to(device)  # (n,)

            perm = torch.argsort(col_values, stable=True)
            perms.append(perm)

            sorted_values = col_values[perm]
            # Find unique patterns and map each column to its unique index
            uniq, inverse = torch.unique(sorted_values, return_inverse=True)

            group_indices.append(inverse)

            # Expand unique pattern integers to k-bit binary rows: (num_unique, k)
            shifts = torch.arange(k - 1, -1, -1, device=device)
            bits = ((uniq.unsqueeze(1) >> shifts) & 1).float()  # (num_unique, k)
            unique_bits.append(bits)

        self.perms = torch.stack(perms)  # (num_blocks, n)
        self.group_indices = group_indices  # list of (n,) tensors
        self.unique_bits = unique_bits  # list of (num_unique, k) tensors

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        n, k = self.n, self.k
        num_blocks = n // k

        # Permute v for all blocks at once: (num_blocks, n)
        v_perm = v[self.perms]

        results = []
        for b in range(num_blocks):
            num_unique = self.unique_bits[b].shape[0]
            # Aggregate permuted v by unique pattern
            aggregated = torch.zeros(num_unique, dtype=v.dtype, device=v.device)
            aggregated.scatter_add_(0, self.group_indices[b], v_perm[b])

            # (num_unique, k).T @ (num_unique,) → (k,)
            result = self.unique_bits[b].t() @ aggregated
            results.append(result)

        return torch.cat(results)
