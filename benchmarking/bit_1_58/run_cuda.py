"""
Benchmark ternary (1.58-bit) CUDA multipliers across n and k values.

Benchmarks PyTorch baselines, official baselines (BitNet, BitBLAS), and the
currently retained fast RSR ternary CUDA variants.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

assert torch.cuda.is_available(), "CUDA not available"

# Fix TORCH_CUDA_ARCH_LIST before any JIT compilation (BitBLAS may set it wrong)
_major, _minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_major}.{_minor}"


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
        times.append(start_ev.elapsed_time(end_ev) / 1000.0)

    return np.median(times)


def _parse_int_list(env_name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    return default if not raw else int(raw)


def fmt_time(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "N/A"
    return f"{t*1e3:.3f}ms"


def load_all_versions():
    """Return list of (label, cls, needs_k) for retained fast CUDA variants."""
    versions = []

    # --- PyTorch baselines (no k) ---
    from multiplier.bit_1_58.pytorch import (
        PytorchBF16Multiplier,
        PytorchFP16Multiplier,
        PytorchINT8Multiplier,
        PytorchMultiplier,
    )

    for label, cls in [
        ("pytorch_fp32", PytorchMultiplier),
        ("pytorch_fp16", PytorchFP16Multiplier),
        ("pytorch_bf16", PytorchBF16Multiplier),
        ("pytorch_int8", PytorchINT8Multiplier),
    ]:

        class _Wrapper:
            def __init__(self, M, _cls=cls):
                self._impl = _cls(M.cuda())

            def __call__(self, v):
                return self._impl(v)

        versions.append((label, _Wrapper, False))

    # --- Official baselines (no k) ---
    try:
        from multiplier.bit_1_58.cuda.bitnet import BitNetCudaOfficialMultiplier

        versions.append(("bitnet", BitNetCudaOfficialMultiplier, False))
    except Exception as e:
        print(f"  [skip bitnet: {e}]")

    try:
        from multiplier.bit_1_58.cuda.bitblas import BitBLASTernaryMultiplier

        versions.append(("bitblas", BitBLASTernaryMultiplier, False))
    except Exception as e:
        print(f"  [skip bitblas: {e}]")

    # --- Retained fast RSR ternary CUDA versions (need k) ---
    rsr_versions = [
        (
            "cuda_v6.3",
            "multiplier.bit_1_58.cuda.rsr_cuda_v6_3",
            "RSRTernaryCudaV6_3Multiplier",
        ),
        (
            "cuda_v6.5",
            "multiplier.bit_1_58.cuda.rsr_cuda_v6_5",
            "RSRTernaryCudaV6_5Multiplier",
        ),
        (
            "cuda_v10.1",
            "multiplier.bit_1_58.cuda.rsr_cuda_v10_1",
            "RSRTernaryCudaV10_1Multiplier",
        ),
        (
            "cuda_adaptive",
            "multiplier.bit_1_58.cuda.rsr_cuda_adaptive",
            "RSRTernaryCudaAdaptiveMultiplier",
        ),
    ]

    import importlib

    for label, mod_name, cls_name in rsr_versions:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            versions.append((label, cls, True))
        except Exception as e:
            print(f"  [skip {label}: {e}]")

    return versions


def random_ternary_matrix(n):
    return torch.randint(-1, 2, (n, n), dtype=torch.float32)


def main():
    n_values = _parse_int_list(
        "RSR_BENCH_N_VALUES", [4096, 8192, 12288, 14336, 16384, 20480, 32768]
    )
    k_values = _parse_int_list("RSR_BENCH_K_VALUES", [2, 4, 6, 8, 10, 12])
    repeats = _parse_int("RSR_BENCH_REPEATS", 20)
    warmup = _parse_int("RSR_BENCH_WARMUP", 5)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repeats: {repeats}, Warmup: {warmup}\n")

    all_versions = load_all_versions()
    baselines = [(lbl, cls) for lbl, cls, needs_k in all_versions if not needs_k]
    rsr_versions = [(lbl, cls) for lbl, cls, needs_k in all_versions if needs_k]

    col_w = 14
    baseline_labels = [lbl for lbl, _ in baselines]
    rsr_labels = [lbl for lbl, _ in rsr_versions]
    all_labels = baseline_labels + rsr_labels

    print(f"  Baselines: {baseline_labels}")
    print(f"  RSR versions: {rsr_labels}\n")

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "results_cuda.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["n", "k"] + all_labels)

    for n in n_values:
        print(f"\n{'='*80}")
        print(f"  n = {n}   (ternary matrix)")
        print(f"{'='*80}")

        M = random_ternary_matrix(n)
        v = torch.randn(n, dtype=torch.float32, device="cuda")

        base_times = []
        for lbl, cls in baselines:
            try:
                m = cls(M)
                t = bench_inference_cuda(m, v, warmup=warmup, repeats=repeats)
            except Exception as e:
                print(f"  [error {lbl} n={n}: {e}]")
                t = float("nan")
            base_times.append(t)

        t_header = f"  {'k':>4}  " + "  ".join(f"{c:>{col_w}}" for c in all_labels)
        print(f"\n  [Inference time — median over {repeats} runs, CUDA events]")
        print(t_header)
        print("  " + "-" * (len(t_header) - 2))

        for k in k_values:
            if n % k == 0:
                rsr_times = []
                for lbl, cls in rsr_versions:
                    try:
                        m = cls(M, k)
                        rsr_times.append(
                            bench_inference_cuda(m, v, warmup=warmup, repeats=repeats)
                        )
                    except Exception as e:
                        print(f"  [error {lbl} n={n} k={k}: {e}]")
                        rsr_times.append(float("nan"))
            else:
                rsr_times = []
                for lbl, cls in rsr_versions:
                    if "adaptive" in lbl:
                        try:
                            m = cls(M, k)
                            rsr_times.append(
                                bench_inference_cuda(
                                    m, v, warmup=warmup, repeats=repeats
                                )
                            )
                        except Exception as e:
                            print(f"  [error {lbl} n={n} k={k}: {e}]")
                            rsr_times.append(float("nan"))
                    else:
                        rsr_times.append(float("nan"))

            all_times = base_times + rsr_times
            valid = [t for t in all_times if not np.isnan(t)]
            best = min(valid) if valid else None

            cells = []
            for t in all_times:
                s = fmt_time(t)
                if best is not None and not np.isnan(t) and abs(t - best) < 1e-9:
                    s = f"*{s}*"
                cells.append(s.rjust(col_w))

            print(f"  {k:>4}  " + "  ".join(cells))

            csv_row = [n, k] + [
                "" if np.isnan(t) else round(t * 1e3, 6) for t in all_times
            ]
            writer.writerow(csv_row)

        print()

    csv_file.close()
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
