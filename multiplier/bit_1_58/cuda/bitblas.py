"""
Official BitBLAS CUDA ternary baseline.

This uses the official `bitblas.Matmul` API for W_INT2 A_INT8 GEMV/GEMM on CUDA.
For this repository's benchmark we use the GEMV case (`M=1`) with exact ternary
weights in {-1, 0, +1} and per-vector int8 activation quantization.
"""

import torch

from multiplier.base import Multiplier

try:
    from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target
    from bitblas.cache import get_database_path, global_operator_cache
except Exception as e:  # pragma: no cover - import-time dependency path
    raise RuntimeError(
        "official BitBLAS package is not available; install `bitblas` to enable this baseline"
    ) from e

_BITBLAS_TARGET = auto_detect_nvidia_target()
_BITBLAS_DATABASE_PATH = get_database_path()
_OPERATOR_CACHE: dict[tuple[int, int], Matmul] = {}
_OPERATOR_ERRORS: dict[tuple[int, int], RuntimeError] = {}


def _get_or_create_operator(n: int, k: int) -> Matmul:
    key = (n, k)
    if key in _OPERATOR_CACHE:
        return _OPERATOR_CACHE[key]
    if key in _OPERATOR_ERRORS:
        raise _OPERATOR_ERRORS[key]

    config = MatmulConfig(
        M=1,
        N=n,
        K=k,
        A_dtype="int8",
        W_dtype="int2",
        accum_dtype="int32",
        out_dtype="float32",
        layout="nt",
        with_bias=False,
        group_size=None,
        with_scaling=False,
        with_zeros=False,
        zeros_mode=None,
    )

    try:
        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(
                _BITBLAS_DATABASE_PATH,
                _BITBLAS_TARGET,
            )

        op = global_operator_cache.get(config)
        if op is None:
            op = Matmul(config, target=_BITBLAS_TARGET, enable_tuning=False)
            global_operator_cache.add(config, op)
        _OPERATOR_CACHE[key] = op
        return op
    except Exception as e:
        err = RuntimeError(
            "official BitBLAS W_INT2A_INT8 baseline is unavailable for "
            f"shape ({n}, {k}) on this machine: {e}"
        )
        _OPERATOR_ERRORS[key] = err
        raise err from e


class BitBLASTernaryMultiplier(Multiplier):
    """Official BitBLAS W_INT2A_INT8 baseline for exact ternary matrices."""

    def __init__(self, M: torch.Tensor):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if M.ndim != 2:
            raise ValueError(f"expected a 2D weight matrix, got shape={tuple(M.shape)}")

        self.out_features, self.in_features = M.shape
        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        self._matmul = _get_or_create_operator(self.out_features, self.in_features)
        weight = self.M.detach().to(torch.int8).cpu().contiguous()
        self._qweight = self._matmul.transform_weight(weight).to(self.device).contiguous()
        del self.M

    @staticmethod
    def _quantize_input(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qmax = 127.0
        scale = qmax / v.abs().amax().clamp(min=1e-5)
        qv = (v * scale).round().clamp(-128, 127).to(torch.int8)
        return qv, scale.reshape(1)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 1 or v.numel() != self.in_features:
            raise ValueError(
                f"expected a vector of shape ({self.in_features},), got {tuple(v.shape)}"
            )

        v_gpu = v.to(self.device, dtype=torch.float32, non_blocking=True)
        qv, scale = self._quantize_input(v_gpu.contiguous())
        out = self._matmul(qv.unsqueeze(0), self._qweight)
        return (out / scale.to(out.device)).squeeze(0)
