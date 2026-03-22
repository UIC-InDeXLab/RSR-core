"""Helpers for building/loading local CUDA shared libraries via nvcc."""

import ctypes
import os
import subprocess

import torch

_CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
_NVCC = os.path.join(_CUDA_HOME, "bin", "nvcc")


def _needs_rebuild(output_path: str, source_paths: list[str]) -> bool:
    if not os.path.exists(output_path):
        return True
    output_mtime = os.path.getmtime(output_path)
    return any(os.path.getmtime(src) > output_mtime for src in source_paths)


def _arch_flag_sets() -> list[list[str]]:
    if not torch.cuda.is_available():
        return [["-gencode=arch=compute_80,code=compute_80"]]

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return [
        [
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            f"-gencode=arch=compute_{arch},code=compute_{arch}",
        ],
        ["-gencode=arch=compute_80,code=compute_80"],
    ]


def build_or_load_cuda_library(
    kernel_dir: str,
    lib_name: str,
    source_names: list[str],
) -> ctypes.CDLL:
    if not os.path.exists(_NVCC):
        raise RuntimeError(f"nvcc not found at {_NVCC}")

    lib_path = os.path.join(kernel_dir, lib_name)
    source_paths = [os.path.join(kernel_dir, name) for name in source_names]

    if _needs_rebuild(lib_path, source_paths):
        last_error = None
        for arch_flags in _arch_flag_sets():
            cmd = [
                _NVCC,
                "-std=c++17",
                "--shared",
                "--compiler-options",
                "-fPIC",
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                *arch_flags,
                *source_paths,
                "-o",
                lib_path,
            ]
            try:
                subprocess.run(
                    cmd,
                    cwd=kernel_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                last_error = None
                break
            except subprocess.CalledProcessError as e:
                last_error = e
        if last_error is not None:
            raise RuntimeError(
                "failed to build CUDA library:\n"
                f"stdout:\n{last_error.stdout}\n"
                f"stderr:\n{last_error.stderr}"
            ) from last_error

    return ctypes.CDLL(lib_path)
