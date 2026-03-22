"""
Official Microsoft BitNet CUDA ternary baseline.

This wraps the official `gpu/bitnet_kernels` W2A8 GEMV kernel from
`microsoft/BitNet`, with the kernel body kept intact and the dispatch table
extended to the square benchmark shapes used in this repository.
"""

import ctypes
import os
import subprocess

import torch

from multiplier.base import Multiplier

from ._bitnet_pack import convert_weight_int8_to_int2

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "kernels",
    "bit_1_58",
    "cuda",
)

_CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
_NVCC = os.path.join(_CUDA_HOME, "bin", "nvcc")
_LIB_PATH = os.path.join(_KERNEL_DIR, "libbitnet.so")

_SUPPORTED_SHAPES = {
    (1024, 1024),
    (2048, 2048),
    (2560, 2560),
    (3200, 3200),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
}


def _needs_rebuild(output_path: str, source_paths: list[str]) -> bool:
    if not os.path.exists(output_path):
        return True
    output_mtime = os.path.getmtime(output_path)
    return any(os.path.getmtime(src) > output_mtime for src in source_paths)


def _build_or_load_library():
    if not os.path.exists(_NVCC):
        raise RuntimeError(
            f"nvcc not found at {_NVCC}. Set CUDA_HOME correctly before importing BitNet CUDA baseline."
        )

    sources = [os.path.join(_KERNEL_DIR, "bitnet_kernels.cu"), os.path.join(_KERNEL_DIR, "bitnet_kernels.h")]
    if _needs_rebuild(_LIB_PATH, sources):
        cmd = [
            _NVCC,
            "-std=c++17",
            "-Xcudafe",
            "--diag_suppress=177",
            "--compiler-options",
            "-fPIC",
            "-lineinfo",
            "--shared",
            os.path.join(_KERNEL_DIR, "bitnet_kernels.cu"),
            "-lcuda",
            "-gencode=arch=compute_80,code=compute_80",
            "-o",
            _LIB_PATH,
        ]
        try:
            subprocess.run(
                cmd,
                cwd=_KERNEL_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "failed to build official BitNet CUDA kernel:\n"
                f"stdout:\n{e.stdout}\n"
                f"stderr:\n{e.stderr}"
            ) from e

    lib = ctypes.CDLL(_LIB_PATH)
    lib.bitlinear_int8xint2.restype = None
    lib.bitlinear_int8xint2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    return lib


_LIB = _build_or_load_library()


class BitNetCudaOfficialMultiplier(Multiplier):
    """Official BitNet GPU W2A8 GEMV baseline for ternary {-1, 0, +1} weights."""

    def __init__(self, M: torch.Tensor):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if M.ndim != 2:
            raise ValueError(f"expected a 2D weight matrix, got shape={tuple(M.shape)}")

        self.out_features, self.in_features = M.shape
        self.shape = (self.out_features, self.in_features)
        if self.shape not in _SUPPORTED_SHAPES:
            supported = ", ".join(f"{n}x{k}" for n, k in sorted(_SUPPORTED_SHAPES))
            raise ValueError(
                "official BitNet CUDA baseline only supports the repository benchmark "
                f"shapes plus native upstream square shapes; got {self.shape}, supported: {supported}"
            )

        self.device = torch.device("cuda")
        super().__init__(M)
        self.prep()

    def prep(self):
        packed_weight = convert_weight_int8_to_int2(self.M)
        self._packed_weight = packed_weight.to(self.device, dtype=torch.int8).contiguous()
        self._weight_scale = torch.ones(1, dtype=torch.bfloat16, device=self.device)
        self._output = torch.empty((1, self.out_features), dtype=torch.bfloat16, device=self.device)
        del self.M

    @staticmethod
    def _quantize_input(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = 127.0 / v.abs().amax().clamp(min=1e-5)
        qv = (v * s).round().clamp(-128, 127).to(torch.int8)
        return qv, s.reshape(1).to(torch.bfloat16)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 1 or v.numel() != self.in_features:
            raise ValueError(
                f"expected a vector of shape ({self.in_features},), got {tuple(v.shape)}"
            )

        v_gpu = v.to(self.device, dtype=torch.float32, non_blocking=True)
        qv, s = self._quantize_input(v_gpu.contiguous())
        stream = torch.cuda.current_stream(device=self.device)
        _LIB.bitlinear_int8xint2(
            ctypes.c_void_p(qv.contiguous().data_ptr()),
            ctypes.c_void_p(self._packed_weight.data_ptr()),
            ctypes.c_void_p(self._output.data_ptr()),
            ctypes.c_void_p(s.contiguous().data_ptr()),
            ctypes.c_void_p(self._weight_scale.data_ptr()),
            ctypes.c_int(1),
            ctypes.c_int(self.out_features),
            ctypes.c_int(self.in_features),
            ctypes.c_void_p(stream.cuda_stream),
        )
        return self._output[0]
