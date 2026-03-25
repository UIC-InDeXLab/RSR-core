"""
Benchmark CPU ternary (1.58-bit) multipliers on a given list of matrix shapes.

Edit SHAPES and K_VALUES below to configure the benchmark.
Timing: median inference latency (preprocessing excluded).
"""

import csv
import importlib
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        multiplier(v)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def fmt(t):
    if t is None or np.isnan(t):
        return "N/A"
    return f"{t * 1e3:.3f}ms"


# ---------------------------------------------------------------------------
# Version discovery
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "rsr_v1_1": "v1.1",
    "rsr_v1_2": "v1.2",
    "rsr_v1_3": "v1.3",
    "rsr_v1_4": "v1.4",
    "rsr_v1_5": "v1.5",
    "rsr_adaptive": "RSR_adaptive",
    "rsr_nonsquare": "RSR",
    "bitnet": "BitNet",
    "tmac": "T-MAC",
}

_EXCLUDE = {"__init__", "base"}


def _stem_to_label(stem):
    if stem in _LABEL_MAP:
        return _LABEL_MAP[stem]
    if stem.startswith("rsr_v"):
        return "v" + stem[len("rsr_v"):].replace("_", ".")
    return stem


def discover_versions():
    versions = []

    from multiplier.bit_1_58.pytorch import PytorchBF16Multiplier

    versions.append(("pytorch", PytorchBF16Multiplier, False))

    cpu_dir = Path(__file__).resolve().parents[2] / "multiplier" / "bit_1_58" / "cpu"
    for p in sorted(cpu_dir.glob("*.py")):
        if p.stem in _EXCLUDE or p.stem.startswith("_"):
            continue
        full = f"multiplier.bit_1_58.cpu.{p.stem}"
        label = _stem_to_label(p.stem)
        try:
            mod = importlib.import_module(full)
        except Exception as e:
            print(f"  [skip {p.stem}: {e}]")
            continue
        cls = next(
            (obj for _, obj in inspect.getmembers(mod, inspect.isclass)
             if obj.__module__ == full and obj.__name__.endswith("Multiplier")),
            None,
        )
        if cls is None:
            continue
        needs_k = "k" in inspect.signature(cls.__init__).parameters
        versions.append((label, cls, needs_k))

    return versions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    versions = discover_versions()
    if METHODS:
        versions = [(l, c, nk) for l, c, nk in versions if l in METHODS]

    baselines = [(l, c) for l, c, nk in versions if not nk]
    rsr_vers = [(l, c) for l, c, nk in versions if nk]
    all_labels = [l for l, _ in baselines] + [l for l, _ in rsr_vers]

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "results_shapes_cpu.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["rows", "cols", "k"] + all_labels)

    col_w = 12

    for rows, cols in SHAPES:
        print(f"\n{'='*80}")
        print(f"  shape = ({rows}, {cols})")
        print(f"{'='*80}")

        M = random_ternary_matrix(rows, cols)
        v = torch.randn(cols, dtype=torch.float32)

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
        print(f"\n  [Inference — median over {REPEATS} runs]")
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
