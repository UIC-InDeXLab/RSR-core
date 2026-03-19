import torch

from multiplier.base import Multiplier


class PytorchMultiplier(Multiplier):
    """Dense FP32 matmul baseline for ternary {-1, 0, +1} matrices."""

    def __init__(self, M: torch.Tensor):
        super().__init__(M)
        self.M = self.M.float()

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.M @ v


class PytorchFP16Multiplier(Multiplier):
    """FP16 matmul baseline for ternary matrices."""

    def __init__(self, M: torch.Tensor):
        super().__init__(M)
        self.M = self.M.half()

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return (self.M @ v.half()).float()


class PytorchBF16Multiplier(Multiplier):
    """BF16 matmul baseline for ternary matrices."""

    def __init__(self, M: torch.Tensor):
        super().__init__(M)
        self.M = self.M.bfloat16()

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return (self.M @ v.bfloat16()).float()


class PytorchINT8Multiplier(Multiplier):
    """INT8 GEMM baseline for ternary matrices.

    Ternary values {-1, 0, +1} fit in int8. Uses torch._int_mm.
    """

    def __init__(self, M: torch.Tensor):
        super().__init__(M)
        self.M = self.M.to(torch.int8)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        n = v.shape[0]
        v_pad = torch.zeros(n, 8, dtype=torch.int8, device=v.device)
        v_pad[:, 0] = v.to(torch.int8)
        result = torch._int_mm(self.M, v_pad)  # (n, 8) int32
        return result[:, 0].float()
