#!/usr/bin/env python3
"""Plot best benchmark time per method and n from results.csv."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def read_best_times(csv_path: Path):
    """
    Return:
      methods: list[str]
      best: dict[method][n] = (best_time_ms, k_for_best)
    """
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        if "n" not in reader.fieldnames or "k" not in reader.fieldnames:
            raise ValueError("CSV must include 'n' and 'k' columns")

        methods = [c for c in reader.fieldnames if c not in ("n", "k")]
        best = defaultdict(dict)

        for row in reader:
            n_str = (row.get("n") or "").strip()
            k_str = (row.get("k") or "").strip()
            if not n_str or not k_str:
                continue

            n = int(n_str)
            k = int(k_str)

            for method in methods:
                raw = (row.get(method) or "").strip()
                if not raw:
                    continue

                time_ms = float(raw)
                prev = best[method].get(n)
                if prev is None or time_ms < prev[0]:
                    best[method][n] = (time_ms, k)

    return methods, best


def parse_ignored_methods(values):
    ignored = set()
    for value in values or []:
        for part in value.split(","):
            name = part.strip()
            if name:
                ignored.add(name)
    return ignored


def filter_methods(methods, ignore_methods=None, warn_unknown=True):
    ignore_methods = set(ignore_methods or [])
    if warn_unknown:
        unknown_ignored = sorted(m for m in ignore_methods if m not in methods)
        if unknown_ignored:
            print(
                "Warning: ignored methods not found in CSV: "
                + ", ".join(unknown_ignored)
            )
    return [m for m in methods if m not in ignore_methods]


def plot_methods(methods, best, out_path: Path, show_plot: bool, title: str):
    if not methods:
        print(f"Warning: no methods selected; skipping plot for {out_path}")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    y_offsets = [-14, -8, -2, 4, 10, 16, 22]
    plotted_methods = []
    for method in methods:
        points = sorted(best[method].items())  # [(n, (time, k)), ...]
        if not points:
            continue
        avg_time_ms = sum(time for _, (time, _) in points) / len(points)
        plotted_methods.append((method, points, avg_time_ms))

    if not plotted_methods:
        plt.close(fig)
        print(f"Warning: no data points found; skipping plot for {out_path}")
        return False

    # Show fastest methods first in legend and plotting order.
    plotted_methods.sort(key=lambda item: (item[2], item[0]))

    for method_idx, (method, points, avg_time_ms) in enumerate(plotted_methods):
        label = f"{method} (avg={avg_time_ms:.3f} ms)"
        x = [n for n, _ in points]
        y = [time for _, (time, _) in points]
        line = ax.plot(x, y, marker="o", linewidth=2, label=label)[0]

        y_offset = y_offsets[method_idx % len(y_offsets)]
        for n, (time_ms, k) in points:
            ax.annotate(
                f"k={k}",
                (n, time_ms),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha="center",
                fontsize=8,
                color=line.get_color(),
            )

    x_values = sorted({n for _, points, _ in plotted_methods for n, _ in points})
    x_formatter = ScalarFormatter()
    x_formatter.set_scientific(False)
    x_formatter.set_useOffset(False)
    # ax.set_xscale("log", base=2)
    ax.set_xticks(x_values)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("Best Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Method")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return True


def plot_best_times(
    csv_path: Path, out_path: Path, show_plot: bool, ignore_methods=None
):
    methods, best = read_best_times(csv_path)
    methods = filter_methods(methods, ignore_methods=ignore_methods, warn_unknown=True)
    return plot_methods(
        methods=methods,
        best=best,
        out_path=out_path,
        show_plot=show_plot,
        title="Best Time per Method Across n (CPU)",
    )


def plot_cuda_only_best_times(csv_path: Path, out_path: Path, ignore_methods=None):
    methods, best = read_best_times(csv_path)
    methods = filter_methods(methods, ignore_methods=ignore_methods, warn_unknown=False)
    has_explicit_cuda = any("cuda" in m.lower() for m in methods)
    is_cuda_csv = "cuda" in csv_path.stem.lower()
    cuda_baseline_prefixes = ("pytorch", "rsr_py", "bitnet", "bitblas")
    cuda_methods = [
        m
        for m in methods
        if "cuda" in m.lower()
        or (
            (has_explicit_cuda or is_cuda_csv)
            and any(m.lower().startswith(p) for p in cuda_baseline_prefixes)
        )
    ]

    if not cuda_methods:
        print("No CUDA methods found after filtering; skipping CUDA-only plot.")
        return False

    return plot_methods(
        methods=cuda_methods,
        best=best,
        out_path=out_path,
        show_plot=False,
        title="Best CUDA Time per Method Across n",
    )


def main():
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Plot best benchmark time per method and n. "
            "For each (method, n), the minimum time over k is used and annotated."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=here / "reports" / "results.csv",
        help="Input CSV path (default: benchmarking/reports/results.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "reports" / "results_best_time_plot.png",
        help="Output image path (default: benchmarking/reports/results_best_time_plot.png)",
    )
    parser.add_argument(
        "--cuda-csv",
        type=Path,
        default=here / "reports" / "results_cuda.csv",
        help=(
            "Input CSV path for CUDA-only plot "
            "(default: benchmarking/reports/results_cuda.csv)"
        ),
    )
    parser.add_argument(
        "--cuda-out",
        type=Path,
        default=None,
        help=(
            "Output image path for CUDA-only plot "
            "(default: <out stem>_cuda_only.png)"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help=(
            "Methods to exclude from both plots. "
            "Use space-separated names (e.g. --ignore pytorch bitnet) or comma-separated."
        ),
    )
    parser.add_argument(
        "--ignore-cpu",
        nargs="*",
        default=["v1.1", "v1.0", "v1.2", "pytorch", "tmac", "v1.3", "v1.4", "v1.5", "pytorch_bf16"],
        help=(
            "Additional methods to exclude only from the main/CPU plot. "
            "Use space-separated names or comma-separated."
        ),
    )
    parser.add_argument(
        "--ignore-cuda",
        nargs="*",
        default=["cuda_v1", "rsr_py", "pytorch_fp32", "pytorch_int8", "v1.1", "v1.0", "v1.2"],
        help=(
            "Additional methods to exclude only from the CUDA-only plot. "
            "Use space-separated names or comma-separated."
        ),
    )
    args = parser.parse_args()

    ignored_common = parse_ignored_methods(args.ignore)
    ignored_cpu = ignored_common | parse_ignored_methods(args.ignore_cpu)
    ignored_cuda = ignored_common | parse_ignored_methods(args.ignore_cuda)

    plot_best_times(args.csv, args.out, args.show, ignore_methods=ignored_cpu)

    cuda_csv = args.cuda_csv
    if not cuda_csv.exists():
        print(f"Warning: CUDA CSV not found at {cuda_csv}; falling back to {args.csv}")
        cuda_csv = args.csv

    cuda_out = args.cuda_out or args.out.with_name(f"{args.out.stem}_cuda_only.png")
    plot_cuda_only_best_times(cuda_csv, cuda_out, ignore_methods=ignored_cuda)


if __name__ == "__main__":
    main()
