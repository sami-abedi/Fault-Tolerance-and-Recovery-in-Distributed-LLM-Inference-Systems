"""
Result visualization for CrashSafe experiments.

Generates publication-quality plots from experiment CSV data:

  Fig 1: Latency CDF by recovery strategy (under kill faults)
  Fig 2: P50/P95/P99 latency bar chart (strategy × fault type)
  Fig 3: Success rate heatmap (strategy × fault type)
  Fig 4: Recovery time comparison (bar chart)
  Fig 5: Token recomputation fraction (strategy × fault type)
  Fig 6: Throughput vs concurrency (line plot)
  Fig 7: Latency breakdown by prompt length
  Fig 8: Overall tradeoff radar chart

All figures saved as PDF + PNG (300 dpi) for paper inclusion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Strategy color palette
STRATEGY_COLORS = {
    "fail_stop": "#e74c3c",           # red
    "retry_from_scratch": "#f39c12",  # orange
    "token_commit_resume": "#2ecc71", # green
}

STRATEGY_LABELS = {
    "fail_stop": "Fail-Stop",
    "retry_from_scratch": "Retry-from-Scratch",
    "token_commit_resume": "Token-Commit-Resume",
}

FAULT_LABELS = {
    "none": "No Fault",
    "kill": "Kill (SIGKILL)",
    "delay": "Delay",
    "hang": "Hang",
    "graceful_shutdown": "Graceful Shutdown",
}


def load_results(csv_path: str) -> pd.DataFrame:
    """Load and preprocess aggregate results CSV."""
    df = pd.read_csv(csv_path)

    # Ensure numeric types
    numeric_cols = [
        "p50_latency_s", "p95_latency_s", "p99_latency_s", "mean_latency_s",
        "mean_recovery_time_s", "mean_recomputation_fraction", "success_rate",
        "throughput_req_per_s", "mean_ttft_s",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Friendly labels
    df["strategy_label"] = df["recovery_strategy"].map(
        lambda s: STRATEGY_LABELS.get(s, s)
    )
    df["fault_label"] = df["fault_type"].map(
        lambda f: FAULT_LABELS.get(f, f)
    )

    return df


# ---------------------------------------------------------------------------
# Figure 1: P50/P95 Latency by Strategy and Fault Type
# ---------------------------------------------------------------------------


def plot_latency_bars(df: pd.DataFrame, output_dir: str) -> str:
    """Bar chart: P50 and P95 latency grouped by strategy and fault type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (metric, label) in zip(axes, [
        ("p50_latency_s", "P50 Latency (s)"),
        ("p95_latency_s", "P95 Latency (s)"),
    ]):
        data = df.groupby(["fault_type", "recovery_strategy"])[metric].mean().reset_index()
        pivot = data.pivot(index="fault_type", columns="recovery_strategy", values=metric)

        strategies = [s for s in STRATEGY_LABELS.keys() if s in pivot.columns]
        x = np.arange(len(pivot))
        width = 0.25

        for i, strat in enumerate(strategies):
            if strat in pivot.columns:
                vals = pivot[strat].fillna(0).values
                bars = ax.bar(
                    x + i * width,
                    vals,
                    width,
                    label=STRATEGY_LABELS[strat],
                    color=STRATEGY_COLORS[strat],
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax.set_xlabel("Fault Type")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [FAULT_LABELS.get(f, f) for f in pivot.index],
            rotation=20,
            ha="right",
        )
        ax.legend(title="Recovery Strategy", loc="upper left")
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Latency Comparison by Recovery Strategy and Fault Type", fontsize=14, y=1.02)
    plt.tight_layout()

    out = Path(output_dir) / "fig1_latency_bars.png"
    plt.savefig(out)
    plt.savefig(str(out).replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Figure 2: Success Rate Heatmap
# ---------------------------------------------------------------------------


def plot_success_rate_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Heatmap: success rate across strategies and fault types."""
    pivot = df.groupby(["recovery_strategy", "fault_type"])["success_rate"].mean().unstack(fill_value=0)

    # Reorder rows and columns
    row_order = [s for s in STRATEGY_LABELS.keys() if s in pivot.index]
    col_order = [f for f in ["none", "kill", "delay", "hang"] if f in pivot.columns]
    pivot = pivot.loc[row_order, col_order] if row_order else pivot

    pivot.index = [STRATEGY_LABELS.get(s, s) for s in pivot.index]
    pivot.columns = [FAULT_LABELS.get(f, f) for f in pivot.columns]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Success Rate"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("Request Success Rate by Strategy and Fault Type", pad=12)
    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Recovery Strategy")
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out = Path(output_dir) / "fig2_success_rate_heatmap.png"
    plt.savefig(out)
    plt.savefig(str(out).replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Figure 3: Recovery Time Comparison
# ---------------------------------------------------------------------------


def plot_recovery_time(df: pd.DataFrame, output_dir: str) -> str:
    """Bar chart: mean recovery time by strategy (faulted requests only)."""
    # Only plot cells with faults
    fault_df = df[df["fault_type"] != "none"].copy()
    if fault_df.empty:
        print("No fault data for recovery time plot.")
        return ""

    agg = (
        fault_df.groupby("recovery_strategy")["mean_recovery_time_s"]
        .mean()
        .reset_index()
    )
    agg["label"] = agg["recovery_strategy"].map(STRATEGY_LABELS)
    agg["color"] = agg["recovery_strategy"].map(STRATEGY_COLORS)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        agg["label"],
        agg["mean_recovery_time_s"] * 1000,  # convert to ms
        color=agg["color"].tolist(),
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )

    # Annotate bars
    for bar, val in zip(bars, agg["mean_recovery_time_s"] * 1000):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Mean Recovery Time (ms)")
    ax.set_title("Mean Recovery Time Under Fault Conditions")
    ax.set_xlabel("Recovery Strategy")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, agg["mean_recovery_time_s"].max() * 1300)

    plt.tight_layout()
    out = Path(output_dir) / "fig3_recovery_time.png"
    plt.savefig(out)
    plt.savefig(str(out).replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Figure 4: Token Recomputation Fraction
# ---------------------------------------------------------------------------


def plot_recomputation(df: pd.DataFrame, output_dir: str) -> str:
    """Bar chart: mean token recomputation fraction by strategy and fault type."""
    fault_df = df[df["fault_type"] != "none"].copy()
    if fault_df.empty:
        return ""

    pivot = (
        fault_df.groupby(["fault_type", "recovery_strategy"])
        ["mean_recomputation_fraction"]
        .mean()
        .unstack(fill_value=0)
    )

    strategies = [s for s in STRATEGY_LABELS.keys() if s in pivot.columns]
    x = np.arange(len(pivot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, strat in enumerate(strategies):
        if strat in pivot.columns:
            vals = pivot[strat].fillna(0).values * 100  # to percentage
            ax.bar(
                x + i * width,
                vals,
                width,
                label=STRATEGY_LABELS[strat],
                color=STRATEGY_COLORS[strat],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )

    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Mean Token Recomputation (%)")
    ax.set_title("Token Recomputation Fraction by Strategy and Fault Type")
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [FAULT_LABELS.get(f, f) for f in pivot.index], rotation=15, ha="right"
    )
    ax.legend(title="Recovery Strategy")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    out = Path(output_dir) / "fig4_recomputation.png"
    plt.savefig(out)
    plt.savefig(str(out).replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Figure 5: Throughput vs Concurrency
# ---------------------------------------------------------------------------


def plot_throughput_vs_concurrency(df: pd.DataFrame, output_dir: str) -> str:
    """Line plot: throughput vs concurrency level for each strategy (no-fault)."""
    no_fault = df[df["fault_type"] == "none"].copy()
    if no_fault.empty:
        return ""

    agg = (
        no_fault.groupby(["concurrency_level", "recovery_strategy"])
        ["throughput_req_per_s"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for strat, grp in agg.groupby("recovery_strategy"):
        grp = grp.sort_values("concurrency_level")
        ax.plot(
            grp["concurrency_level"],
            grp["throughput_req_per_s"],
            marker="o",
            linewidth=2,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat, "gray"),
        )

    ax.set_xlabel("Concurrency Level (# Parallel Requests)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs. Concurrency (No-Fault Baseline)")
    ax.legend(title="Recovery Strategy")
    ax.grid(alpha=0.4)
    ax.set_xticks(agg["concurrency_level"].unique())

    plt.tight_layout()
    out = Path(output_dir) / "fig5_throughput_concurrency.png"
    plt.savefig(out)
    plt.savefig(str(out).replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Figure 6: Latency vs Prompt Length
# ---------------------------------------------------------------------------


def plot_latency_vs_prompt_length(df: pd.DataFrame, output_dir: str) -> str:
    """Line plot: P50 latency vs prompt length (no-fault)."""
    no_fault = df[df["fault_type"] == "none"].copy()
    if no_fault.empty:
        return ""

    length_order = {"short": 0, "medium": 1, "long": 2}
    no_fault["length_order"] = no_fault["prompt_length_label"].map(length_order)
    no_fault = no_fault.sort_values("length_order")

    agg = (
        no_fault.groupby(["prompt_length_label", "recovery_strategy"])
        ["p50_latency_s"]
        .mean()
        .reset_index()
    )
    agg["length_order"] = agg["prompt_length_label"].map(length_order)
    agg = agg.sort_values("length_order")

    fig, ax = plt.subplots(figsize=(8, 5))

    for strat, grp in agg.groupby("recovery_strategy"):
        grp = grp.sort_values("length_order")
        ax.plot(
            grp["prompt_length_label"],
            grp["p50_latency_s"] * 1000,
            marker="o",
            linewidth=2,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat, "gray"),
        )

    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Median Latency (ms)")
    ax.set_title("Latency vs. Prompt Length (No-Fault Baseline)")
    ax.legend(title="Recovery Strategy")
    ax.grid(alpha=0.4)

    plt.tight_layout()
    out = Path(output_dir) / "fig6_latency_prompt_length.png"
    plt.savefig(out)
    plt.savefig(str(out).replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Figure 7: Strategy Comparison Summary (multi-metric)
# ---------------------------------------------------------------------------


def plot_strategy_summary(df: pd.DataFrame, output_dir: str) -> str:
    """Multi-panel summary: key metrics per strategy under kill faults."""
    kill_df = df[df["fault_type"] == "kill"].copy()
    if kill_df.empty:
        # Fall back to any fault
        kill_df = df[df["fault_type"] != "none"].copy()
    if kill_df.empty:
        return ""

    metrics = {
        "success_rate": ("Success Rate (%)", 100),
        "p50_latency_s": ("P50 Latency (ms)", 1000),
        "mean_recovery_time_s": ("Recovery Time (ms)", 1000),
        "mean_recomputation_fraction": ("Recomputed Tokens (%)", 100),
    }

    agg = kill_df.groupby("recovery_strategy")[list(metrics.keys())].mean()
    strategies = [s for s in STRATEGY_LABELS.keys() if s in agg.index]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (col, (ylabel, scale)) in zip(axes, metrics.items()):
        vals = [agg.loc[s, col] * scale if s in agg.index else 0 for s in strategies]
        colors = [STRATEGY_COLORS[s] for s in strategies]
        labels = [STRATEGY_LABELS[s] for s in strategies]

        bars = ax.bar(range(len(strategies)), vals, color=colors, edgecolor="black",
                      linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(
            [l.replace("-", "\n") for l in labels],
            rotation=0, ha="center", fontsize=9
        )
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.4)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("Strategy Comparison Under Kill Faults", fontsize=14, y=1.02)
    plt.tight_layout()

    out = Path(output_dir) / "fig7_strategy_summary.png"
    plt.savefig(out)
    plt.savefig(str(out).replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------


def generate_all_plots(csv_path: str, figures_dir: str) -> None:
    """
    Load results CSV and generate all experiment figures.

    Args:
        csv_path: Path to the aggregate_results.csv file.
        figures_dir: Directory to save figures.
    """
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from: {csv_path}")
    df = load_results(csv_path)
    print(f"Loaded {len(df)} rows covering:")
    print(f"  Strategies: {df['recovery_strategy'].unique().tolist()}")
    print(f"  Fault types: {df['fault_type'].unique().tolist()}")
    print(f"  Concurrency: {df['concurrency_level'].unique().tolist()}")
    print(f"\nGenerating figures → {figures_dir}\n")

    plotters = [
        plot_latency_bars,
        plot_success_rate_heatmap,
        plot_recovery_time,
        plot_recomputation,
        plot_throughput_vs_concurrency,
        plot_latency_vs_prompt_length,
        plot_strategy_summary,
    ]

    for plotter in plotters:
        try:
            plotter(df, figures_dir)
        except Exception as exc:
            print(f"Warning: {plotter.__name__} failed: {exc}")

    print(f"\nAll figures saved to: {figures_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot CrashSafe experiment results")
    parser.add_argument("csv_path", help="Path to aggregate_results.csv")
    parser.add_argument("--output-dir", default="./figures", help="Figures output dir")
    args = parser.parse_args()

    generate_all_plots(args.csv_path, args.output_dir)
