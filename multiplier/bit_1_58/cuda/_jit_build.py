"""Helper to JIT-compile CUDA kernels with proper arch detection."""

import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import _get_build_directory, load


def _ensure_cuda_arch():
    """Set TORCH_CUDA_ARCH_LIST using current GPU. Always overwrite since
    some libraries (e.g. BitBLAS) may set it to an incorrect value."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


_ensure_cuda_arch()


def _ensure_venv_bin_on_path():
    """Make helper executables installed in the current venv visible.

    We often invoke scripts as ``./venv/bin/python`` without activating the
    environment, so PATH may not include the matching ``venv/bin`` directory.
    PyTorch's extension loader looks for ``ninja`` on PATH; prepend the current
    interpreter's bin directory so JIT builds work consistently.
    """
    bindir = os.path.dirname(sys.executable)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bindir and bindir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([bindir, *path_entries]) if path_entries else bindir


_ensure_venv_bin_on_path()

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "kernels", "bit_1_58", "cuda"
)


def _wait_for_or_clear_stale_lock(name: str) -> None:
    """Avoid indefinite waits on stale torch extension lock files.

    PyTorch's extension loader waits forever when a lock file is left behind in
    the build cache. For benchmarks we want bounded behavior: wait briefly for a
    real concurrent build, then either clear a stale lock (if a compiled module
    already exists) or raise a clear error so callers can skip the variant.
    """
    build_dir = Path(_get_build_directory(name, verbose=False))
    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return

    timeout_s = float(os.environ.get("RSR_CUDA_JIT_LOCK_TIMEOUT_S", "10"))
    poll_s = 0.1
    deadline = time.monotonic() + timeout_s

    while lock_path.exists() and time.monotonic() < deadline:
        time.sleep(poll_s)

    if not lock_path.exists():
        return

    shared_objects = list(build_dir.glob(f"{name}*.so"))
    if shared_objects:
        try:
            lock_path.unlink()
            return
        except FileNotFoundError:
            return

    raise RuntimeError(
        "timed out waiting for CUDA JIT build lock "
        f"{lock_path}; remove the stale lock or increase "
        "RSR_CUDA_JIT_LOCK_TIMEOUT_S if another build is still running"
    )


def load_kernel(name: str, source_file: str) -> object:
    """JIT-compile and load a CUDA kernel module."""
    _wait_for_or_clear_stale_lock(name)
    return load(
        name=name,
        sources=[os.path.join(_KERNEL_DIR, source_file)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
