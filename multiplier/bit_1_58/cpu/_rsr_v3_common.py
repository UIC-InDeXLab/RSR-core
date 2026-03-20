"""Shared helpers for ternary RSR v3.x CPU wrappers."""

import ctypes

import torch


FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
INT8_PTR = ctypes.POINTER(ctypes.c_int8)
INT32_PTR = ctypes.POINTER(ctypes.c_int32)
UINT16_PTR = ctypes.POINTER(ctypes.c_uint16)


def ensure_cpu_float32_contiguous(v: torch.Tensor) -> torch.Tensor:
    """Return a CPU float32 contiguous view suitable for the C kernels."""
    v_detached = v.detach()
    if (
        v_detached.device.type == "cpu"
        and v_detached.dtype == torch.float32
        and v_detached.is_contiguous()
    ):
        return v_detached
    return v_detached.to(device="cpu", dtype=torch.float32).contiguous()


def tensor_float_ptr(t: torch.Tensor):
    return ctypes.cast(t.data_ptr(), FLOAT_PTR)
