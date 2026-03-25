"""
Plot CPU bit-1 benchmark results: best time per method across k values.

x-axis: matrix size n (square: n×n)
y-axis: median inference time (ms)
lines:  one per method
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV = Path(__file__).parent / "reports" / "results_shapes_cpu.csv"
OUTPUT = Path(__file__).parent / "reports" / "plot_shapes_cpu.png"

# Rename method labels for display; unlisted methods use their original name
LABEL_RENAME = {
    # "old_name": "display name",
    "pytorch": "PyTorch (bf16)",
    "bitnet": "BitNet",
    "tmac": "T-MAC",
    "RSR": r"RSR$^*$",
}

# Methods that don't use k — skip the k annotation for these
NO_K_ANNOTATION = {"pytorch", "bitnet", "tmac", "PyTorch", "BitNet", "T-MAC"}


def main():
    df = pd.read_csv(CSV)

    method_cols = [c for c in df.columns if c not in ("rows", "cols", "k")]

    # For each (shape, method) pick the best (minimum) time across all k values
    best = df.groupby(["rows", "cols"])[method_cols].min().reset_index()
    best["n"] = best["rows"]  # shapes are square

    # For each (shape, method) record which k achieved the minimum time
    best_k = df.groupby(["rows", "cols"]).apply(
        lambda g: {
            m: g.loc[g[m].idxmin(), "k"] if g[m].notna().any() else None
            for m in method_cols
        }
    )

    # Sort methods by mean time across shapes (best first)
    method_cols = sorted(method_cols, key=lambda m: best[m].mean())

    fig, ax = plt.subplots(figsize=(6, 4))

    RSR_METHODS = {"RSR"}
    BASELINE_MARKERS = ["s", "^", "D", "v", "P", "X", "h"]
    BASELINE_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#76b7b2", "#ff9da7"]

    baseline_idx = 0
    for method in method_cols:
        sub = best[["n", method]].dropna()
        if sub.empty:
            continue
        is_rsr = method in RSR_METHODS
        if is_rsr:
            marker, color, lw, ms, alpha, zorder = "*", "crimson", 3, 12, 1.0, 3
        else:
            marker = BASELINE_MARKERS[baseline_idx % len(BASELINE_MARKERS)]
            color = BASELINE_COLORS[baseline_idx % len(BASELINE_COLORS)]
            baseline_idx += 1
            lw, ms, alpha, zorder = 1.2, 6, 0.5, 2
        (line,) = ax.plot(
            sub["n"], sub[method],
            marker=marker,
            markersize=ms,
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=zorder,
            label=LABEL_RENAME.get(method, method),
        )
        for _, row in sub.iterrows():
            shape_key = (int(row["n"]), int(row["n"]))
            k_val = best_k.get(shape_key, {}).get(method)
            if k_val is not None and method not in NO_K_ANNOTATION:
                ax.annotate(
                    f"k={int(k_val)}",
                    xy=(row["n"], row[method]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color=line.get_color(),
                )

    ax.set_xlabel("Matrix size n  (n × n)", fontsize=16)
    # ax.set_ylabel("Time (ms)", fontsize=16)
    ax.set_title("Multiplication Time (ms) [CPU 1.58-bit]", fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    # ax.set_xscale("log", base=2)
    # ax.set_yscale("log")

    plt.tight_layout()
    out = OUTPUT
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
