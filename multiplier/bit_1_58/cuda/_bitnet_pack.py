"""
Official Microsoft BitNet GPU weight packing for int2 weights.

Derived from:
https://github.com/microsoft/BitNet/blob/01eb415772c342d9f20dc42772f1583ae1e5b102/gpu/pack_weight.py
"""

import numpy as np
import torch


def _b_global_16x32_to_shared_load_16x32_layout(i: int, j: int) -> tuple[int, int]:
    thread_id = i * 2 + j // 16
    row = (thread_id // 16) * 8 + (thread_id % 8)
    col = (j % 16) + 16 * ((thread_id % 16) // 8)
    return row, col


def _permutate_weight_fastest(weight: np.ndarray) -> np.ndarray:
    wmma_n = 16
    wmma_k = 32
    n = weight.shape[0]
    k = weight.shape[1]

    mapping = np.zeros((wmma_n, wmma_k, 2), dtype=int)
    for ii in range(wmma_n):
        for jj in range(wmma_k):
            mapping[ii, jj] = _b_global_16x32_to_shared_load_16x32_layout(ii, jj)

    i_indices = np.arange(n // wmma_n)[:, np.newaxis, np.newaxis, np.newaxis]
    j_indices = np.arange(k // wmma_k)[np.newaxis, :, np.newaxis, np.newaxis]
    src_i = i_indices * wmma_n + mapping[:, :, 0]
    src_j = j_indices * wmma_k + mapping[:, :, 1]
    return weight[src_i, src_j]


def _compress_int2_to_int8(int2_weight: np.ndarray) -> np.ndarray:
    int8_weight = np.zeros(
        (*int2_weight.shape[:-1], int2_weight.shape[-1] // 4),
        dtype=np.int8,
    )
    for j in range(int2_weight.shape[-1] // 4):
        for k in range(4):
            int8_weight[:, :, :, j] |= int2_weight[:, :, :, j * 4 + k] << (k * 2)
    return int8_weight


def _interleave_weight_int8(qweight: np.ndarray, nbits: int = 2) -> np.ndarray:
    qweight = qweight.view(np.int32)
    new_qweight = np.zeros_like(qweight)
    bits_stride = 8
    mask = (1 << nbits) - 1
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits

    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift
    return new_qweight.view(np.int8)


def convert_weight_int8_to_int2(weight: torch.Tensor) -> torch.Tensor:
    """
    Pack a ternary {-1, 0, +1} matrix into the official BitNet GPU int2 layout.

    The official CUDA kernel decodes values by subtracting 2 from the stored
    2-bit codes, so {-1, 0, +1} becomes {1, 2, 3}.
    """

    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight matrix, got shape={tuple(weight.shape)}")

    n, k = weight.shape
    if n % 16 != 0 or k % 32 != 0:
        raise ValueError(
            f"official BitNet CUDA packing requires N % 16 == 0 and K % 32 == 0, got {(n, k)}"
        )

    weight_np = weight.detach().cpu().to(torch.int8).numpy().copy()
    weight_np = weight_np + 2
    permutated_weight = _permutate_weight_fastest(weight_np)
    compressed_weight = _compress_int2_to_int8(permutated_weight)
    interleaved_weight = _interleave_weight_int8(compressed_weight, 2)

    return torch.from_numpy(interleaved_weight.reshape(n, k // 4))
