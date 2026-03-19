"""
Benchmark inference time of ternary (1.58-bit) CPU multipliers across n and k values.

Timing: measures median inference latency (preprocessing excluded).
"""

import csv
import importlib
import inspect
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


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


# ---------------------------------------------------------------------------
# Version discovery
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "rsr_v1_1": "v1.1",
    "rsr_v1_2": "v1.2",
    "rsr_v1_3": "v1.3",
    "rsr_v1_4": "v1.4",
    "rsr_v1_5": "v1.5",
    "rsr_adaptive": "adaptive",
}

_CPU_MODULE_EXCLUDE = {"__init__", "base"}


def _stem_to_label(stem: str) -> str:
    if stem in _LABEL_MAP:
        return _LABEL_MAP[stem]
    if stem.startswith("rsr_v"):
        suffix = stem[len("rsr_v"):]
        return f"v{suffix.replace('_', '.')}"
    return stem


def discover_all_versions():
    """Return list of (label, cls, needs_k) for pytorch + CPU multiplier modules."""
    versions = []

    # Baselines
    from multiplier.bit_1_58.pytorch import PytorchMultiplier, PytorchFP16Multiplier, PytorchBF16Multiplier
    versions.append(("pytorch", PytorchMultiplier, False))
    versions.append(("pytorch_fp16", PytorchFP16Multiplier, False))
    versions.append(("pytorch_bf16", PytorchBF16Multiplier, False))

    try:
        from multiplier.bit_1_58.cpu.bitnet import BitNetTernaryMultiplier
        versions.append(("bitnet", BitNetTernaryMultiplier, False))
    except Exception as e:
        print(f"  [skip bitnet: {e}]")

    # RSR Python (v1.0)
    try:
        from multiplier.bit_1_58.rsr_py import RSRTernaryV1_0Multiplier
        versions.append(("v1.0", RSRTernaryV1_0Multiplier, True))
    except Exception as e:
        print(f"  [skip v1.0: {e}]")

    # CPU versions
    cpu_dir = Path(__file__).resolve().parents[2] / "multiplier" / "bit_1_58" / "cpu"
    stems = sorted(
        p.stem
        for p in cpu_dir.glob("*.py")
        if p.stem not in _CPU_MODULE_EXCLUDE and not p.stem.startswith("_")
    )

    for stem in stems:
        full = f"multiplier.bit_1_58.cpu.{stem}"
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


def random_ternary_matrix(n):
    return torch.randint(-1, 2, (n, n), dtype=torch.float32)


def main():
    n_values = [1024, 2048, 4096, 8192, 8192 * 2]
    k_values = [2, 4, 6, 8, 10, 12]
    repeats = 20
    warmup = 5

    all_versions = discover_all_versions()
    baselines = [(lbl, cls) for lbl, cls, needs_k in all_versions if not needs_k]
    rsr_versions = [(lbl, cls) for lbl, cls, needs_k in all_versions if needs_k]

    col_w = 12
    baseline_labels = [lbl for lbl, _ in baselines]
    rsr_labels = [lbl for lbl, _ in rsr_versions]
    all_labels = baseline_labels + rsr_labels

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "results.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["n", "k"] + all_labels)

    for n in n_values:
        print(f"\n{'='*80}")
        print(f"  n = {n}   (ternary matrix)")
        print(f"{'='*80}")

        M = random_ternary_matrix(n)
        v = torch.randn(n, dtype=torch.float32)

        # Benchmark baselines
        base_times = []
        for lbl, cls in baselines:
            try:
                m = cls(M)
                t = bench_inference(m, v, warmup=warmup, repeats=repeats)
            except Exception as e:
                print(f"  [error {lbl} n={n}: {e}]")
                t = float("nan")
            base_times.append(t)

        t_header = f"  {'k':>4}  " + "  ".join(f"{c:>{col_w}}" for c in all_labels)
        print(f"\n  [Inference time — median over {repeats} runs]")
        print(t_header)
        print("  " + "-" * (len(t_header) - 2))

        for k in k_values:
            if n % k == 0:
                filtered_base_times = base_times[:]
                rsr_times = []
                for lbl, cls in rsr_versions:
                    try:
                        m = cls(M, k)
                        rsr_times.append(
                            bench_inference(m, v, warmup=warmup, repeats=repeats)
                        )
                    except Exception as e:
                        print(f"  [error {lbl} n={n} k={k}: {e}]")
                        rsr_times.append(float("nan"))
            else:
                filtered_base_times = [
                    t if lbl in ("pytorch", "pytorch_fp16", "pytorch_bf16") else float("nan")
                    for (lbl, _), t in zip(baselines, base_times)
                ]
                rsr_times = []
                for lbl, cls in rsr_versions:
                    if lbl == "adaptive":
                        try:
                            m = cls(M, k)
                            rsr_times.append(
                                bench_inference(m, v, warmup=warmup, repeats=repeats)
                            )
                        except Exception as e:
                            print(f"  [error {lbl} n={n} k={k}: {e}]")
                            rsr_times.append(float("nan"))
                    else:
                        rsr_times.append(float("nan"))

            all_times = filtered_base_times + rsr_times
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
