"""
Find the best RSR group size (k) for each ternary matrix shape on CPU or CUDA.

Benchmarks RSR non-square multiplier at various k values against baselines
(PyTorch FP32, BF16) for every layer shape found in preprocessed models.

Outputs:
  - Console table with timing and speedups
  - CSV with full results:  reports/best_k_{device}.csv
  - JSON with best k per shape: reports/best_k_{device}.json

Usage:
    python -m benchmarking.bit_1_58.bench_best_k --device cpu
    python -m benchmarking.bit_1_58.bench_best_k --device cuda
    python -m benchmarking.bit_1_58.bench_best_k --device cpu --shapes 2560x2560 4096x14336
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multiplier.bit_1_58.pytorch import (
    PytorchBF16Multiplier,
    PytorchMultiplier,
)

# All unique layer shapes found across preprocessed models.
# Discovered from integrations/hf/preprocessed/*/rsr_config.json.
ALL_SHAPES = [
    # BitNet-b1.58-2B-4T
    (640, 2560),
    (2560, 2560),
    (2560, 6912),
    (6912, 2560),
    # Llama3-8B-1.58-100B-tokens
    (1024, 4096),
    (4096, 4096),
    (4096, 14336),
    (14336, 4096),
    # Falcon3-10B-Instruct-1.58bit
    (1024, 3072),
    (3072, 3072),
    (3072, 23040),
    (23040, 3072),
]

K_VALUES = [2, 4, 6, 8, 10, 12]


def random_ternary(n_rows, n_cols):
    return torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.float32)


def bench_inference(fn, warmup=5, repeats=20):
    """Return median latency in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def bench_cpu(shapes, k_values, warmup, repeats):
    from multiplier.bit_1_58.cpu.rsr_v3_3_nonsquare import (
        RSRTernaryV3_3NonSquareMultiplier,
    )

    rows = []
    for n_rows, n_cols in shapes:
        M = random_ternary(n_rows, n_cols)
        v = torch.randn(n_cols, dtype=torch.float32)

        t_fp32 = bench_inference(
            lambda: PytorchMultiplier(M)(v) if False else None,
            warmup=0, repeats=1,
        )
        # Proper baselines
        pt = PytorchMultiplier(M)
        t_fp32 = bench_inference(lambda: pt(v), warmup=warmup, repeats=repeats)

        M_bf = M.bfloat16()
        v_bf = v.bfloat16()
        pt_bf = PytorchBF16Multiplier(M)
        t_bf16 = bench_inference(lambda: pt_bf(v), warmup=warmup, repeats=repeats)

        print(f"\n  Shape ({n_rows:>5}, {n_cols:>5})  "
              f"FP32={t_fp32*1e3:.3f}ms  BF16={t_bf16*1e3:.3f}ms")

        best_k, best_t = None, float("inf")
        for k in k_values:
            try:
                rsr = RSRTernaryV3_3NonSquareMultiplier(M, k)
                t_rsr = bench_inference(
                    lambda: rsr(v), warmup=warmup, repeats=repeats,
                )
            except Exception as e:
                print(f"    k={k:>2}  FAILED: {e}")
                continue

            speedup_fp32 = t_fp32 / t_rsr
            speedup_bf16 = t_bf16 / t_rsr
            tag = ""
            if t_rsr < best_t:
                best_t = t_rsr
                best_k = k
                tag = " <-- best"
            print(f"    k={k:>2}  RSR={t_rsr*1e3:.3f}ms  "
                  f"vs FP32={speedup_fp32:.2f}x  vs BF16={speedup_bf16:.2f}x{tag}")

            rows.append({
                "n_rows": n_rows, "n_cols": n_cols, "k": k,
                "rsr_ms": round(t_rsr * 1e3, 4),
                "fp32_ms": round(t_fp32 * 1e3, 4),
                "bf16_ms": round(t_bf16 * 1e3, 4),
                "speedup_vs_fp32": round(speedup_fp32, 3),
                "speedup_vs_bf16": round(speedup_bf16, 3),
            })

    return rows


def bench_cuda_inference(fn, warmup=5, repeats=20):
    """Return median latency in seconds using CUDA events."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return float(np.median(times)) / 1e3  # return seconds


def bench_cuda(shapes, k_values, warmup, repeats):
    from multiplier.bit_1_58.cuda.rsr_cuda_v6_3_nonsquare import (
        RSRTernaryCudaV6_3NonSquareMultiplier,
    )

    rows = []
    for n_rows, n_cols in shapes:
        M = random_ternary(n_rows, n_cols)
        v = torch.randn(n_cols, dtype=torch.float32)

        # FP32 baseline on CUDA
        M_cuda = M.cuda()
        v_cuda = v.cuda()

        def pt_fp32():
            torch.mv(M_cuda, v_cuda)

        t_fp32 = bench_cuda_inference(pt_fp32, warmup=warmup, repeats=repeats)

        # BF16 baseline on CUDA
        M_bf_cuda = M_cuda.bfloat16()
        v_bf_cuda = v_cuda.bfloat16()

        def pt_bf16():
            torch.mv(M_bf_cuda, v_bf_cuda)

        t_bf16 = bench_cuda_inference(pt_bf16, warmup=warmup, repeats=repeats)

        print(f"\n  Shape ({n_rows:>5}, {n_cols:>5})  "
              f"FP32={t_fp32*1e3:.3f}ms  BF16={t_bf16*1e3:.3f}ms")

        best_k, best_t = None, float("inf")
        for k in k_values:
            try:
                rsr = RSRTernaryCudaV6_3NonSquareMultiplier(M, k)
                v_dev = v.cuda()

                def rsr_fn():
                    rsr(v_dev)

                t_rsr = bench_cuda_inference(rsr_fn, warmup=warmup, repeats=repeats)
            except Exception as e:
                print(f"    k={k:>2}  FAILED: {e}")
                continue

            speedup_fp32 = t_fp32 / t_rsr
            speedup_bf16 = t_bf16 / t_rsr
            tag = ""
            if t_rsr < best_t:
                best_t = t_rsr
                best_k = k
                tag = " <-- best"
            print(f"    k={k:>2}  RSR={t_rsr*1e3:.3f}ms  "
                  f"vs FP32={speedup_fp32:.2f}x  vs BF16={speedup_bf16:.2f}x{tag}")

            rows.append({
                "n_rows": n_rows, "n_cols": n_cols, "k": k,
                "rsr_ms": round(t_rsr * 1e3, 4),
                "fp32_ms": round(t_fp32 * 1e3, 4),
                "bf16_ms": round(t_bf16 * 1e3, 4),
                "speedup_vs_fp32": round(speedup_fp32, 3),
                "speedup_vs_bf16": round(speedup_bf16, 3),
            })

    return rows


def parse_shape(s):
    """Parse 'NxM' or 'N,M' into (int, int)."""
    for sep in ("x", "X", ","):
        if sep in s:
            parts = s.split(sep)
            return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Cannot parse shape: {s!r}. Use NxM format.")


def main():
    parser = argparse.ArgumentParser(
        description="Find best RSR k per matrix shape.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--shapes", nargs="+", default=None,
                        help="Override shapes (NxM format). Default: all preprocessed model shapes.")
    parser.add_argument("--k-values", nargs="+", type=int, default=K_VALUES,
                        help="k values to test")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    shapes = [parse_shape(s) for s in args.shapes] if args.shapes else ALL_SHAPES

    print(f"Device: {args.device}")
    print(f"Shapes: {len(shapes)}")
    print(f"k values: {args.k_values}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")

    if args.device == "cpu":
        rows = bench_cpu(shapes, args.k_values, args.warmup, args.repeats)
    else:
        rows = bench_cuda(shapes, args.k_values, args.warmup, args.repeats)

    if not rows:
        print("\nNo results.")
        return

    # --- Save CSV ---
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / f"best_k_{args.device}.csv"
    fieldnames = ["n_rows", "n_cols", "k", "rsr_ms", "fp32_ms", "bf16_ms",
                  "speedup_vs_fp32", "speedup_vs_bf16"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV saved to {csv_path}")

    # --- Compute best k per shape ---
    best_map = {}
    for r in rows:
        key = f"{r['n_rows']}x{r['n_cols']}"
        if key not in best_map or r["rsr_ms"] < best_map[key]["rsr_ms"]:
            best_map[key] = {
                "k": r["k"],
                "rsr_ms": r["rsr_ms"],
                "fp32_ms": r["fp32_ms"],
                "bf16_ms": r["bf16_ms"],
                "speedup_vs_fp32": r["speedup_vs_fp32"],
                "speedup_vs_bf16": r["speedup_vs_bf16"],
            }

    json_path = reports_dir / f"best_k_{args.device}.json"
    with open(json_path, "w") as f:
        json.dump(best_map, f, indent=2)
    print(f"Best-k JSON saved to {json_path}")

    # --- Summary table ---
    print(f"\n{'='*78}")
    print(f"  BEST k PER SHAPE  (device={args.device})")
    print(f"{'='*78}")
    print(f"  {'Shape':>14}  {'k':>3}  {'RSR':>9}  {'FP32':>9}  {'BF16':>9}  {'vs FP32':>8}  {'vs BF16':>8}")
    print(f"  {'-'*72}")
    for key in sorted(best_map, key=lambda k: tuple(int(x) for x in k.split("x"))):
        b = best_map[key]
        print(f"  {key:>14}  {b['k']:>3}  {b['rsr_ms']:>8.3f}ms  {b['fp32_ms']:>8.3f}ms"
              f"  {b['bf16_ms']:>8.3f}ms  {b['speedup_vs_fp32']:>7.2f}x  {b['speedup_vs_bf16']:>7.2f}x")


if __name__ == "__main__":
    main()
