"""Runtime RSR CUDA multiplier that operates on preprocessed (saved) tensors.

This mirrors ``RSRTernaryCudaV2_0Multiplier`` but loads directly from
safetensors artifacts instead of running the prep step on a raw weight matrix.
"""

from __future__ import annotations

from typing import Any

import torch

# Tensor keys expected in the safetensors file for CUDA layers
CUDA_TENSOR_KEYS = ("perms", "group_packed", "block_meta")


def _load_rsr_cuda_module():
    from ._jit_build import load_kernel
    return load_kernel("rsr_ternary_cuda_v2_0", "rsr_ternary_v2_0.cu")


_RSR_CUDA_MODULE = None


def _get_cuda_module():
    global _RSR_CUDA_MODULE
    if _RSR_CUDA_MODULE is None:
        _RSR_CUDA_MODULE = _load_rsr_cuda_module()
    return _RSR_CUDA_MODULE


class RSRPreprocessedCudaMultiplier:
    """CUDA GEMV runtime built from saved v2.0 compact metadata."""

    def __init__(
        self,
        layer_name: str,
        layer_meta: dict[str, Any],
        tensors: dict[str, torch.Tensor],
    ):
        self.layer_name = layer_name
        self.n_rows = int(layer_meta["n_rows"])
        self.n_cols = int(layer_meta["n_cols"])
        self.k = int(layer_meta["k"])
        self.n_rows_padded = int(
            layer_meta.get(
                "n_rows_padded",
                ((self.n_rows + self.k - 1) // self.k) * self.k,
            )
        )
        self._num_blocks = int(
            layer_meta.get("num_blocks", self.n_rows_padded // self.k)
        )
        self.device = torch.device("cuda")

        self._perms = tensors["perms"].to(dtype=torch.uint16, device=self.device)
        self._group_packed = tensors["group_packed"].to(
            dtype=torch.int64,
            device=self.device,
        )
        self._block_meta = tensors["block_meta"].to(
            dtype=torch.int32,
            device=self.device,
        )
        self._out = torch.empty(
            self.n_rows_padded, dtype=torch.float32, device=self.device
        )

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 1 or vector.shape[0] != self.n_cols:
            raise ValueError(
                f"Layer {self.layer_name!r} expected vector of shape ({self.n_cols},), "
                f"got {tuple(vector.shape)}"
            )

        if vector.device != self.device or vector.dtype != torch.float32:
            v_gpu = vector.to(self.device, dtype=torch.float32)
        else:
            v_gpu = vector

        _get_cuda_module().rsr_ternary_gemv_v2_0(
            self._perms,
            self._group_packed,
            self._block_meta,
            v_gpu.contiguous(),
            self._out,
            self.n_cols,
            self.k,
            self._num_blocks,
        )
        return self._out[: self.n_rows]
