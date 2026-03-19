import torch

from ..base import Multiplier
from .rsr_cpp_v4_2 import RSRCppV4_2Multiplier


class RSRAdaptiveMultiplier(Multiplier):
    """
    RSR multiplier that supports any positive k by zero-padding M to n_pad x n_pad,
    where n_pad is the smallest multiple of k >= n.
    """

    def __init__(self, M: torch.Tensor, k: int):
        assert k > 0, "k must be positive"
        assert M.ndim == 2, "M must be a 2D tensor"
        assert M.shape[0] == M.shape[1], "Matrix must be square"

        self.n = M.shape[0]
        self.k = k
        self.n_padded = ((self.n + k - 1) // k) * k
        self.pad = self.n_padded - self.n
        super().__init__(M)
        self.prep()

    def prep(self):
        if self.pad == 0:
            M_padded = self.M
        else:
            M_padded = torch.zeros(
                (self.n_padded, self.n_padded),
                dtype=self.M.dtype,
                device=self.M.device,
            )
            M_padded[: self.n, : self.n] = self.M

        self._inner = RSRCppV4_2Multiplier(M_padded, self.k)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        assert v.ndim == 1, "v must be a 1D tensor"
        assert (
            v.shape[0] == self.n
        ), f"Expected vector length {self.n}, got {v.shape[0]}"

        if self.pad == 0:
            v_padded = v
        else:
            v_padded = torch.zeros(self.n_padded, dtype=v.dtype, device=v.device)
            v_padded[: self.n] = v

        out_padded = self._inner(v_padded)
        return out_padded[: self.n]
