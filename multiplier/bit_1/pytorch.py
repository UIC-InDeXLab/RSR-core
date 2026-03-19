import torch

from .base import Multiplier


class PytorchMultiplier(Multiplier):
    def prep(self):
        self.M = self.M

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.M @ v


class PytorchFP16Multiplier(Multiplier):
    """FP16 matmul baseline — uses tensor cores on CUDA, 2x memory reduction."""

    def __init__(self, M: torch.Tensor):
        super().__init__(M)
        self.prep()

    def prep(self):
        self.M = self.M.half()

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return (self.M @ v.half()).float()


class PytorchINT8Multiplier(Multiplier):
    """INT8 GEMM baseline via torch._int_mm (cublasLtMatmul INT8 path).

    Stores binary M as int8. Uses integer matmul for the multiply,
    then converts back to float. For binary {0,1} matrices this is exact.
    """

    def __init__(self, M: torch.Tensor):
        super().__init__(M)
        self.prep()

    def prep(self):
        self.M = self.M.to(torch.int8)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        # torch._int_mm requires 2nd operand columns to be a multiple of 8
        n = v.shape[0]
        v_pad = torch.zeros(n, 8, dtype=torch.int8, device=v.device)
        v_pad[:, 0] = v.to(torch.int8)
        result = torch._int_mm(self.M, v_pad)  # (n, 8) int32
        return result[:, 0].float()


class PytorchBF16Multiplier(Multiplier):
    """BF16 matmul baseline — tensor cores, same exponent range as FP32."""

    def __init__(self, M: torch.Tensor):
        super().__init__(M)
        self.prep()

    def prep(self):
        self.M = self.M.bfloat16()

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return (self.M @ v.bfloat16()).float()
