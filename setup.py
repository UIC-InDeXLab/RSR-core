"""Custom build: compile CPU and CUDA kernels during pip install."""

import subprocess
import sys
import os
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

ROOT = os.path.dirname(os.path.abspath(__file__))

CPU_KERNEL_DIRS = [
    os.path.join(ROOT, "kernels", "bit_1", "cpu"),
    os.path.join(ROOT, "kernels", "bit_1_58", "cpu"),
]

CUDA_KERNEL_DIR_BIT1 = os.path.join(ROOT, "kernels", "bit_1", "cuda")
CUDA_KERNEL_DIR_BIT158 = os.path.join(ROOT, "kernels", "bit_1_58", "cuda")


def _build_cpu_kernels():
    for d in CPU_KERNEL_DIRS:
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "Makefile")):
            subprocess.check_call(["make", "-C", d])


def _print_cuda_skip_warning():
    """Print a warning that CUDA kernels were not pre-built."""
    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    print()
    print(f"{YELLOW}setup.py: CUDA not available — CUDA kernels were not pre-built.{RESET}")
    print(f"{YELLOW}          They will be JIT-compiled on the first CUDA run, if available.{RESET}")
    print()
    print(f"  {BOLD_RED}FOR BENCHMARKS PAY ATTENTION TO FIRST BUILD TIME{RESET}")
    print()


def _build_cuda_kernels():
    """JIT-compile all CUDA kernels so first run has zero compilation delay."""
    try:
        import torch
        if not torch.cuda.is_available():
            _print_cuda_skip_warning()
            return
    except ImportError:
        _print_cuda_skip_warning()
        return

    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    # Ensure ninja is on PATH
    bindir = os.path.dirname(sys.executable)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bindir and bindir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([bindir, *path_entries])

    # -- bit_1 CUDA kernels (torch JIT) --
    bit1_kernels = [
        ("rsr_cuda_v5_9", "rsr_v5_9.cu"),
        ("rsr_cuda_v5_8", "rsr_v5_8.cu"),
        ("rsr_cuda_v5_6", "rsr_v5_6.cu"),
        ("rsr_cuda_v4_10", "rsr_v4_10.cu"),
    ]
    for name, source in bit1_kernels:
        source_path = os.path.join(CUDA_KERNEL_DIR_BIT1, source)
        if not os.path.isfile(source_path):
            continue
        print(f"setup.py: JIT compiling {name} ...")
        try:
            load(
                name=name,
                sources=[source_path],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        except Exception as e:
            print(f"setup.py: WARNING: failed to compile {name}: {e}")

    # -- bit_1_58 CUDA kernels (torch JIT) --
    bit158_jit_kernels = [
        ("rsr_ternary_cuda_v2_0", "rsr_ternary_v2_0.cu"),
    ]
    for name, source in bit158_jit_kernels:
        source_path = os.path.join(CUDA_KERNEL_DIR_BIT158, source)
        if not os.path.isfile(source_path):
            continue
        print(f"setup.py: JIT compiling {name} ...")
        try:
            load(
                name=name,
                sources=[source_path],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        except Exception as e:
            print(f"setup.py: WARNING: failed to compile {name}: {e}")

    # -- bit_1_58 BitNet kernel (nvcc direct) --
    bitnet_source = os.path.join(CUDA_KERNEL_DIR_BIT158, "bitnet_kernels.cu")
    bitnet_lib = os.path.join(CUDA_KERNEL_DIR_BIT158, "libbitnet.so")
    if os.path.isfile(bitnet_source) and not os.path.isfile(bitnet_lib):
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(nvcc):
            arch = f"{major}{minor}"
            cmd = [
                nvcc, "-std=c++17", "--shared", "--compiler-options", "-fPIC",
                "-O3", "--use_fast_math", "-lineinfo",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
                f"-gencode=arch=compute_{arch},code=compute_{arch}",
                bitnet_source, "-o", bitnet_lib,
            ]
            print(f"setup.py: compiling libbitnet.so ...")
            try:
                subprocess.run(cmd, cwd=CUDA_KERNEL_DIR_BIT158, check=True,
                               capture_output=True, text=True)
            except Exception as e:
                print(f"setup.py: WARNING: failed to compile libbitnet.so: {e}")


def _build_all_kernels():
    _build_cpu_kernels()
    _build_cuda_kernels()


class BuildPyWithKernels(build_py):
    def run(self):
        _build_all_kernels()
        super().run()


class DevelopWithKernels(develop):
    def run(self):
        _build_all_kernels()
        super().run()


setup(
    cmdclass={
        "build_py": BuildPyWithKernels,
        "develop": DevelopWithKernels,
    },
)
