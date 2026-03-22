"""
Benchmark inference time of non-square ternary matrix-vector multiply.

Compares:
  - pytorch: dense torch matmul (baseline)
  - rsr_ternary_nonsquare: RSR ternary v3.3 non-square multiplier

Timing: measures median inference latency (preprocessing excluded).
"""

import csv
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multiplier.bit_1_58.pytorch import PytorchMultiplier
from multiplier.bit_1_58.cpu.rsr_v3_3_nonsquare import RSRTernaryV3_3NonSquareMultiplier


def bench_inference(multiplier, v, warmup=5, repeats=20):
    """Time only the __call__ (inference), return median in seconds."""
    for _ in range(warmup):
        multiplier(v)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        multiplier(v)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def fmt_time(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "N/A"
    return f"{t*1e3:.3f}ms"


def random_ternary(n_rows, n_cols):
    return torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.float32)


def main():
    # Non-square shapes: (n_rows, n_cols) — representative of neural network layers
    shapes = [
        (1024, 4096),
        (4096, 1024),
        (2048, 2048),
        (4096, 4096),
        (4096, 11008),
        (11008, 4096),
        (8192, 8192),
        (4096, 14336),
        (14336, 4096),
    ]
    k_values = [4, 8, 12]
    repeats = 20
    warmup = 5

    labels = ["pytorch", "rsr_ternary_nonsquare"]
    col_w = 22

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "results_nonsquare.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["n_rows", "n_cols", "k"] + labels)

    for n_rows, n_cols in shapes:
        print(f"\n{'='*80}")
        print(f"  shape = ({n_rows}, {n_cols})")
        print(f"{'='*80}")

        M = random_ternary(n_rows, n_cols)
        v = torch.randn(n_cols, dtype=torch.float32)

        # Pytorch baseline (same for all k)
        try:
            m_pt = PytorchMultiplier(M)
            t_pytorch = bench_inference(m_pt, v, warmup=warmup, repeats=repeats)
        except Exception as e:
            print(f"  [error pytorch: {e}]")
            t_pytorch = float("nan")

        header = f"  {'k':>4}  " + "  ".join(f"{c:>{col_w}}" for c in labels)
        print(f"\n  [Inference time — median over {repeats} runs]")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for k in k_values:
            try:
                m_rsr = RSRTernaryV3_3NonSquareMultiplier(M, k)
                t_rsr = bench_inference(m_rsr, v, warmup=warmup, repeats=repeats)
            except Exception as e:
                print(f"  [error rsr_ternary_nonsquare k={k}: {e}]")
                t_rsr = float("nan")

            all_times = [t_pytorch, t_rsr]
            valid = [t for t in all_times if not np.isnan(t)]
            best = min(valid) if valid else None

            cells = []
            for t in all_times:
                s = fmt_time(t)
                if best is not None and not np.isnan(t) and abs(t - best) < 1e-9:
                    s = f"*{s}*"
                cells.append(s.rjust(col_w))

            print(f"  {k:>4}  " + "  ".join(cells))

            csv_row = [n_rows, n_cols, k] + [
                "" if np.isnan(t) else round(t * 1e3, 6) for t in all_times
            ]
            writer.writerow(csv_row)

        print()

    csv_file.close()
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
