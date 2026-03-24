"""
Benchmark inference time of non-square ternary matrix-vector multiply on CUDA.

Compares:
  - pytorch_fp32: dense torch matmul on CUDA (baseline)
  - pytorch_fp16: FP16 matmul on CUDA
  - rsr_ternary_nonsquare: RSR v2.0 ternary multiplier

Timing: uses CUDA events for accurate GPU measurement; reports median
inference latency (preprocessing excluded).
"""

import csv
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

assert torch.cuda.is_available(), "CUDA not available"

from multiplier.bit_1_58.pytorch import PytorchMultiplier, PytorchFP16Multiplier
from multiplier.bit_1_58.cuda.rsr_cuda_v2_0 import RSRTernaryCudaV2_0Multiplier


def bench_inference_cuda(multiplier, v, warmup=5, repeats=20):
    """Time __call__ using CUDA events, return median in seconds."""
    for _ in range(warmup):
        multiplier(v)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start_ev.record()
        multiplier(v)
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev) / 1000.0)  # ms -> sec

    return np.median(times)


def fmt_time(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "N/A"
    return f"{t*1e3:.3f}ms"


def random_ternary(n_rows, n_cols):
    return torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.float32)


def main():
    # Non-square shapes representative of neural network layers
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

    labels = ["pytorch_fp32", "pytorch_fp16", "rsr_ternary_nonsquare"]
    col_w = 22

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repeats: {repeats}, Warmup: {warmup}\n")

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "results_cuda_nonsquare.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["n_rows", "n_cols", "k"] + labels)

    for n_rows, n_cols in shapes:
        print(f"\n{'='*80}")
        print(f"  shape = ({n_rows}, {n_cols})")
        print(f"{'='*80}")

        M = random_ternary(n_rows, n_cols)
        v = torch.randn(n_cols, dtype=torch.float32, device="cuda")

        # Pytorch baselines (same for all k)
        try:
            m_fp32 = PytorchMultiplier(M.cuda())
            t_fp32 = bench_inference_cuda(m_fp32, v, warmup=warmup, repeats=repeats)
        except Exception as e:
            print(f"  [error pytorch_fp32: {e}]")
            t_fp32 = float("nan")

        try:
            m_fp16 = PytorchFP16Multiplier(M.cuda())
            t_fp16 = bench_inference_cuda(m_fp16, v, warmup=warmup, repeats=repeats)
        except Exception as e:
            print(f"  [error pytorch_fp16: {e}]")
            t_fp16 = float("nan")

        header = f"  {'k':>4}  " + "  ".join(f"{c:>{col_w}}" for c in labels)
        print(f"\n  [Inference time — median over {repeats} runs, CUDA events]")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for k in k_values:
            try:
                m_rsr = RSRTernaryCudaV2_0Multiplier(M, k)
                t_rsr = bench_inference_cuda(m_rsr, v, warmup=warmup, repeats=repeats)
            except Exception as e:
                print(f"  [error rsr_ternary_nonsquare k={k}: {e}]")
                t_rsr = float("nan")

            all_times = [t_fp32, t_fp16, t_rsr]
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
