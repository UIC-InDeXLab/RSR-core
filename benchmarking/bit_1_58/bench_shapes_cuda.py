"""
Benchmark CUDA ternary (1.58-bit) multipliers on a given list of matrix shapes.

Edit SHAPES and K_VALUES below to configure the benchmark.
Timing: CUDA events, median inference latency (preprocessing excluded).
"""

import csv
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

assert torch.cuda.is_available(), "CUDA not available"

_major, _minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_major}.{_minor}"

# ---------------------------------------------------------------------------
# Configure here
# ---------------------------------------------------------------------------

SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
    (32768, 32768),
]

K_VALUES = [2, 4, 6, 8, 10]

# Limit to these method labels; empty list = all discovered methods
METHODS = ["BitNet", "RSR", "pytorch"]

REPEATS = 30
WARMUP = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_ternary_matrix(rows, cols):
    return torch.randint(-1, 2, (rows, cols), dtype=torch.float32)


def bench(multiplier, v, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        multiplier(v)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        multiplier(v)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return np.median(times)


def fmt(t):
    if t is None or np.isnan(t):
        return "N/A"
    return f"{t * 1e3:.3f}ms"


# ---------------------------------------------------------------------------
# Version loading
# ---------------------------------------------------------------------------


def load_versions():
    versions = []

    from multiplier.bit_1_58.pytorch import (
        PytorchMultiplier, PytorchFP16Multiplier,
        PytorchBF16Multiplier, PytorchINT8Multiplier,
    )
    for label, cls in [
        ("pytorch", PytorchBF16Multiplier),
        ("pytorch_fp32", PytorchMultiplier),
        ("pytorch_fp16", PytorchFP16Multiplier),
        ("pytorch_bf16", PytorchBF16Multiplier),
        ("pytorch_int8", PytorchINT8Multiplier),
    ]:
        class _W:
            def __init__(self, M, _cls=cls):
                self._impl = _cls(M.cuda())
            def __call__(self, v):
                return self._impl(v)
        versions.append((label, _W, False))

    try:
        from multiplier.bit_1_58.cuda.bitnet import BitNetCudaOfficialMultiplier
        versions.append(("BitNet", BitNetCudaOfficialMultiplier, False))
    except Exception as e:
        print(f"  [skip bitnet: {e}]")

    try:
        from multiplier.bit_1_58.cuda.bitblas import BitBLASTernaryMultiplier
        versions.append(("bitblas", BitBLASTernaryMultiplier, False))
    except Exception as e:
        print(f"  [skip bitblas: {e}]")

    try:
        mod = importlib.import_module("multiplier.bit_1_58.cuda.rsr_cuda_v2_0")
        cls = getattr(mod, "RSRTernaryCudaV2_0Multiplier")
        versions.append(("RSR", cls, True))
    except Exception as e:
        print(f"  [skip cuda_v2.0: {e}]")

    return versions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repeats: {REPEATS}, Warmup: {WARMUP}\n")

    versions = load_versions()
    if METHODS:
        versions = [(l, c, nk) for l, c, nk in versions if l in METHODS]

    baselines = [(l, c) for l, c, nk in versions if not nk]
    rsr_vers = [(l, c) for l, c, nk in versions if nk]
    all_labels = [l for l, _ in baselines] + [l for l, _ in rsr_vers]

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "results_shapes_cuda.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["rows", "cols", "k"] + all_labels)

    col_w = 14

    for rows, cols in SHAPES:
        print(f"\n{'='*80}")
        print(f"  shape = ({rows}, {cols})")
        print(f"{'='*80}")

        M = random_ternary_matrix(rows, cols)
        v = torch.randn(cols, dtype=torch.float32, device="cuda")

        base_times = []
        for lbl, cls in baselines:
            try:
                m = cls(M)
                t = bench(m, v)
            except Exception as e:
                print(f"  [error {lbl}: {e}]")
                t = float("nan")
            base_times.append(t)

        header = f"  {'k':>4}  " + "  ".join(f"{c:>{col_w}}" for c in all_labels)
        print(f"\n  [Inference — median over {REPEATS} runs, CUDA events]")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for k in K_VALUES:
            rsr_times = []
            for lbl, cls in rsr_vers:
                if rows % k != 0:
                    rsr_times.append(float("nan"))
                    continue
                try:
                    m = cls(M, k)
                    rsr_times.append(bench(m, v))
                except Exception as e:
                    print(f"  [error {lbl} k={k}: {e}]")
                    rsr_times.append(float("nan"))

            all_times = base_times + rsr_times
            valid = [t for t in all_times if not np.isnan(t)]
            best = min(valid) if valid else None

            cells = []
            for t in all_times:
                s = fmt(t)
                if best is not None and not np.isnan(t) and abs(t - best) < 1e-9:
                    s = f"*{s}*"
                cells.append(s.rjust(col_w))

            print(f"  {k:>4}  " + "  ".join(cells))
            writer.writerow(
                [rows, cols, k] + ["" if np.isnan(t) else round(t * 1e3, 6) for t in all_times]
            )
            csv_file.flush()

        print()

    csv_file.close()
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
