"""Custom build: compile CPU kernels (make) during pip install."""

import subprocess
import os
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

ROOT = os.path.dirname(os.path.abspath(__file__))

KERNEL_DIRS = [
    os.path.join(ROOT, "kernels", "bit_1", "cpu"),
    os.path.join(ROOT, "kernels", "bit_1_58", "cpu"),
]


def _build_kernels():
    for d in KERNEL_DIRS:
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "Makefile")):
            subprocess.check_call(["make", "-C", d])


class BuildPyWithKernels(build_py):
    def run(self):
        _build_kernels()
        super().run()


class DevelopWithKernels(develop):
    def run(self):
        _build_kernels()
        super().run()


setup(
    cmdclass={
        "build_py": BuildPyWithKernels,
        "develop": DevelopWithKernels,
    },
)
