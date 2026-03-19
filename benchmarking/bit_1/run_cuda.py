"""
Benchmark inference time of all CUDA multipliers across n and k values.

Auto-discovers all multiplier classes from multiplier/cuda/ and includes
baselines (pytorch on CUDA).  Timing uses CUDA events for accurate GPU
measurement; reports median inference latency (preprocessing excluded).
"""

import csv
import importlib
import inspect
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

assert torch.cuda.is_available(), "CUDA not available"

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


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
        times.append(start_ev.elapsed_time(end_ev) / 1000.0)  # ms → sec

    return np.median(times)


def fmt_time(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "N/A"
    return f"{t*1e3:.3f}ms"


# ---------------------------------------------------------------------------
# Version discovery
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "rsr_cuda_v1": "cuda_v1",
    "rsr_cuda_v2": "cuda_v2",
    "rsr_cuda_v5": "cuda_v5",
    "rsr_cuda_adaptive": "cuda_adaptive",
}


def _stem_to_label(stem: str) -> str:
    if stem in _LABEL_MAP:
        return _LABEL_MAP[stem]
    if stem.startswith("rsr_cuda_v"):
        suffix = stem[len("rsr_cuda_v"):]
        return f"cuda_v{suffix.replace('_', '.')}"
    return stem


def discover_all_versions():
    """Return list of (label, cls, needs_k) for baselines + all cuda modules."""
    versions = []

    # --- Baselines (k-independent) ---

    from multiplier.bit_1.pytorch import (
        PytorchMultiplier,
        PytorchFP16Multiplier,
        PytorchINT8Multiplier,
        PytorchBF16Multiplier,
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

    # RSR Python on CUDA (if available)
    try:
        from multiplier.bit_1.rsr_py import RSRPythonMultiplier

        class RSRPyCudaMultiplier(RSRPythonMultiplier):
            """rsr_py.py with tensors on GPU."""
            def __init__(self, M, k):
                super().__init__(M.cuda(), k)

        versions.append(("rsr_py", RSRPyCudaMultiplier, True))
    except Exception as e:
        print(f"  [skip rsr_py baseline: {e}]")

    # --- CUDA modules (auto-discovered) ---
    cuda_dir = Path(__file__).resolve().parents[2] / "multiplier" / "bit_1" / "cuda"
    stems = sorted(
        p.stem
        for p in cuda_dir.glob("*.py")
        if p.stem not in ("__init__",) and not p.stem.startswith("_")
    )

    for stem in stems:
        full = f"multiplier.bit_1.cuda.{stem}"
        label = _stem_to_label(stem)

        try:
            mod = importlib.import_module(full)
        except Exception as e:
            print(f"  [skip {stem}: import error: {e}]")
            continue

        cls = None
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == full and obj.__name__.endswith("Multiplier"):
                cls = obj
                break

        if cls is None:
            print(f"  [skip {stem}: class not found]")
            continue

        needs_k = "k" in inspect.signature(cls.__init__).parameters
        versions.append((label, cls, needs_k))

    return versions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def random_binary_matrix(n):
    return torch.randint(0, 2, (n, n), dtype=torch.float32)


def main():
    n_values = [1024, 2048, 4096, 8192, 16384]
    k_values = [2, 4, 6, 8, 10, 12]
    repeats = 20
    warmup = 5

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repeats: {repeats}, Warmup: {warmup}\n")

    all_versions = discover_all_versions()
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
        print(
            f"  n = {n}   (matrix: {n*n*4 / 1024**2:.1f} MB float32)"
        )
        print(f"{'='*80}")

        M = random_binary_matrix(n)
        v = torch.randn(n, dtype=torch.float32, device="cuda")

        # Benchmark baselines (same time for every k row)
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
                # All versions can run
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
                # Only baselines and adaptive versions can run
                rsr_times = []
                for lbl, cls in rsr_versions:
                    if "adaptive" in lbl:
                        try:
                            m = cls(M, k)
                            rsr_times.append(
                                bench_inference_cuda(m, v, warmup=warmup, repeats=repeats)
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
