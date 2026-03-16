"""
Generate the full consolidated plot suite from saved Assignment-2 result tables.

All plots use a consistent seaborn 'whitegrid' + 'talk' context theme so that
they look clean and readable when embedded in a LaTeX report at 300 dpi.

Usage
-----
    python3 generate_all_plots.py
    python3 generate_all_plots.py --results_dir results --out_dir results/all_plots
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ── Global theme ──────────────────────────────────────────────────────────────
# Set once here so every plot in this module inherits the same look.
sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
PALETTE      = sns.color_palette("colorblind")
MODEL_ORDER  = ["ResNet50", "EfficientNet-B0", "ConvNeXt-Tiny"]
MODEL_COLORS = {m: PALETTE[i] for i, m in enumerate(MODEL_ORDER)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(results_dir: str, relative_path: str) -> pd.DataFrame:
    """Load a CSV from *results_dir/relative_path*, raising a clear error if missing."""
    full_path = os.path.join(results_dir, relative_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Required CSV not found: {full_path}\n"
            "Run the corresponding scenario script first."
        )
    return pd.read_csv(full_path)


def _save(fig: plt.Figure, output_path: str, dpi: int = 300) -> None:
    """Save *fig* at *dpi* and close it."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _sorted_models(dataframe: pd.DataFrame) -> list[str]:
    """Return models in MODEL_ORDER, keeping any extras at the end."""
    available = dataframe["Model"].dropna().unique().tolist()
    ordered   = [m for m in MODEL_ORDER if m in available]
    ordered  += [m for m in available if m not in ordered]
    return ordered


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=14, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    sns.despine(ax=ax)


def _humanize_label(label: str) -> str:
    text = str(label).replace("_", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wrap_label(label: str, width: int = 12) -> str:
    return textwrap.fill(_humanize_label(label), width=width, break_long_words=False)


def _wrap_display_label(label: str, width: int = 12) -> str:
    return textwrap.fill(str(label), width=width, break_long_words=False)


def _short_model_label(label: str) -> str:
    mapping = {
        "ConvNeXt-Tiny": "ConvNeXt-\nTiny",
        "EfficientNet-B0": "EffNet-B0",
        "ResNet50": "ResNet50",
    }
    return mapping.get(str(label), str(label))


def _legend_outside(fig: plt.Figure, ax: plt.Axes, title: str) -> None:
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        title=title,
    )
    fig.subplots_adjust(right=0.80)


# ── Scenario 4.1 — Linear probe summary plots ────────────────────────────────

def plot_s41(s41: pd.DataFrame, output_dir: str) -> None:
    """Generate three summary plots for Scenario 4.1 (linear probe transfer)."""
    model_order = _sorted_models(s41)
    s41 = s41.set_index("Model").loc[model_order].reset_index()

    # --- Best validation accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.barplot(
        data=s41,
        x="Model",
        y="Best Val Acc (%)",
        hue="Model",
        dodge=False,
        palette=MODEL_COLORS,
        legend=False,
        ax=ax,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(75, 100)
    for bar, val in zip(ax.patches, s41["Best Val Acc (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    _style(ax, "Linear probe: best validation accuracy",
           xlabel="Backbone", ylabel="Best validation accuracy")
    _save(fig, os.path.join(output_dir, "s41_best_val_accuracy.png"))

    # --- Accuracy vs parameters trade-off (direct labels, FLOPs in annotation) ---
    s41["Params_M"] = s41["Total Params"] / 1e6
    s41["FLOPs_G"]  = s41["FLOPs"] / 1e9
    s41["BubbleSize"] = 220 + s41["FLOPs_G"] * 55
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.scatter(
        s41["Params_M"],
        s41["Best Val Acc (%)"],
        s=s41["BubbleSize"],
        c=[MODEL_COLORS.get(m, PALETTE[0]) for m in s41["Model"]],
        alpha=0.90,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )
    offsets = {
        "EfficientNet-B0": (10, 6),
        "ResNet50": (10, 6),
        "ConvNeXt-Tiny": (10, -2),
    }
    for _, row in s41.iterrows():
        dx, dy = offsets.get(row["Model"], (8, 6))
        label = f"{row['Model']}\n{row['Params_M']:.1f}M params | {row['FLOPs_G']:.2f}G FLOPs"
        ax.annotate(
            label,
            (row["Params_M"], row["Best Val Acc (%)"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92),
        )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlim(2, 31)
    ax.set_ylim(85.8, 95.4)
    _style(ax, "Linear probe efficiency trade-off",
           xlabel="Total parameters (millions)",
           ylabel="Best validation accuracy")
    _save(fig, os.path.join(output_dir, "s41_efficiency_tradeoff.png"))

    # --- Train-val gap bar chart ---
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.barplot(data=s41, x="Model", y="Train-Val Gap (%)",
                palette=[MODEL_COLORS.get(m, PALETTE[0]) for m in model_order],
                hue="Model", legend=False, ax=ax)
    ax.set_ylim(0, max(s41["Train-Val Gap (%)"].max() * 1.35, 10))
    _style(ax, "Overfitting signal: train–validation gap",
           xlabel="Backbone", ylabel="Train − val accuracy gap (pp)")
    _save(fig, os.path.join(output_dir, "s41_train_val_gap.png"))


# ── Scenario 4.2 — Fine-tuning strategy plots ────────────────────────────────

def plot_s42(s42: pd.DataFrame, output_dir: str) -> None:
    """Generate summary plots for Scenario 4.2 (fine-tuning strategies)."""
    strategy_order = ["linear_probe", "selective", "last_block", "full"]
    model_order    = _sorted_models(s42)

    # Enforce categorical ordering
    s42 = s42.copy()
    s42["Strategy"] = pd.Categorical(s42["Strategy"],
                                     categories=strategy_order, ordered=True)
    s42 = s42.sort_values(["Model", "Strategy"])

    # --- Heatmap: accuracy per model × strategy ---
    pivot = (s42.pivot(index="Model", columns="Strategy", values="Best Val Acc (%)")
               .reindex(model_order))
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlGnBu",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Best val. accuracy (%)"},
        ax=ax,
    )
    ax.set_title("Fine-tuning strategies: accuracy heatmap", fontsize=14, pad=10)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Backbone", fontsize=12)
    _save(fig, os.path.join(output_dir, "s42_strategy_heatmap.png"))

    # --- Line: accuracy vs unfrozen backbone % ---
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    sns.lineplot(
        data=s42,
        x="Unfrozen (%)",
        y="Best Val Acc (%)",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        linewidth=2.3,
        markersize=8,
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _style(ax, "Accuracy vs unfrozen backbone fraction",
           xlabel="Unfrozen backbone parameters (%)",
           ylabel="Best validation accuracy")
    _legend_outside(fig, ax, title="Backbone")
    _save(fig, os.path.join(output_dir, "s42_unfrozen_vs_accuracy.png"))

    # --- Bar: train-val gap by strategy ---
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    sns.barplot(data=s42, x="Strategy", y="Train-Val Gap (%)",
                hue="Model", palette=MODEL_COLORS,
                order=strategy_order, ax=ax)
    ax.set_xticks(range(len(strategy_order)))
    ax.set_xticklabels([_wrap_label(label, width=10) for label in strategy_order])
    _legend_outside(fig, ax, title="Backbone")
    _style(ax, "Overfitting gap by fine-tuning strategy",
           xlabel="Strategy", ylabel="Train − val gap (pp)")
    _save(fig, os.path.join(output_dir, "s42_gap_by_strategy.png"))

    # --- Scatter: accuracy vs log(trainable params) — parameter efficiency ---
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    sns.scatterplot(
        data=s42,
        x=s42["Trainable Params"] / 1e6,
        y="Best Val Acc (%)",
        hue="Model",
        style="Model",
        s=110,
        palette=MODEL_COLORS,
        ax=ax,
    )
    for _, row in s42.iterrows():
        ax.annotate(row["Strategy"],
                    (row["Trainable Params"] / 1e6, row["Best Val Acc (%)"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _legend_outside(fig, ax, title="Backbone")
    _style(ax, "Parameter efficiency: accuracy vs tuned parameters",
           xlabel="Trainable parameters (millions, log scale)",
           ylabel="Best validation accuracy")
    _save(fig, os.path.join(output_dir, "s42_accuracy_vs_tuned_params_summary.png"))


# ── Scenario 4.3 — Few-shot learning plots ───────────────────────────────────

def plot_s43(s43: pd.DataFrame, output_dir: str) -> None:
    """Generate summary plots for Scenario 4.3 (few-shot learning)."""
    fraction_to_numeric = {"5%": 5.0, "20%": 20.0, "50%": 50.0, "100%": 100.0}
    s43 = s43.copy()
    s43["Fraction_num"] = s43["Data Fraction"].map(fraction_to_numeric)
    s43 = s43.sort_values(["Model", "Fraction_num"])

    # --- Line: accuracy vs data fraction ---
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.lineplot(
        data=s43,
        x="Fraction_num",
        y="Best Val Acc (%)",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        linewidth=2.3,
        markersize=8,
        palette=MODEL_COLORS,
        ax=ax,
    )
    x_ticks = sorted(s43["Fraction_num"].dropna().unique())
    ax.set_xticks(x_ticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _legend_outside(fig, ax, title="Backbone")
    _style(ax, "Sample efficiency: accuracy vs training data fraction",
           xlabel="Training data used (%)",
           ylabel="Best validation accuracy")
    _save(fig, os.path.join(output_dir, "s43_accuracy_vs_fraction_from_table.png"))

    # --- Line: train-val gap vs fraction (overfitting signal) ---
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.lineplot(
        data=s43,
        x="Fraction_num",
        y="Train-Val Gap (%)",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        linewidth=2.3,
        markersize=8,
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.set_xticks(x_ticks)
    _legend_outside(fig, ax, title="Backbone")
    _style(ax, "Overfitting audit: train–validation gap vs data fraction",
           xlabel="Training data used (%)",
           ylabel="Train − val gap (pp)")
    _save(fig, os.path.join(output_dir, "s43_gap_vs_fraction.png"))

    # --- Bar: relative performance drop Δ at 5% ---
    five_pct_rows = s43[s43["Data Fraction"] == "5%"].copy()
    five_pct_rows["Relative Drop Δ (%)"] = pd.to_numeric(
        five_pct_rows["Relative Drop Δ (%)"], errors="coerce"
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.barplot(
        data=five_pct_rows,
        x="Model",
        y="Relative Drop Δ (%)",
        hue="Model",
        dodge=False,
        palette=MODEL_COLORS,
        legend=False,
        ax=ax,
    )
    for bar, val in zip(ax.patches, five_pct_rows["Relative Drop Δ (%)"]):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    _style(ax, "Data efficiency: relative accuracy drop Δ (100% → 5% data)",
           xlabel="Backbone", ylabel="Relative drop Δ (%)")
    _save(fig, os.path.join(output_dir, "s43_relative_drop_delta.png"))


# ── Scenario 4.4 — Corruption robustness plots ───────────────────────────────

def plot_s44(s44: pd.DataFrame, output_dir: str) -> None:
    """Generate summary plots for Scenario 4.4 (corruption robustness)."""
    corruption_order = [
        "Gaussian σ=0.05", "Gaussian σ=0.1", "Gaussian σ=0.2",
        "MotionBlur k=15", "Brightness f=1.5",
    ]
    s44 = s44.copy()
    s44["Corruption"] = pd.Categorical(
        s44["Corruption"], categories=corruption_order, ordered=True
    )
    s44 = s44.sort_values(["Corruption", "Model"])

    # --- Grouped bar: corrupted accuracy ---
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    sns.barplot(data=s44, x="Corruption", y="Corrupted Acc (%)",
                hue="Model", palette=MODEL_COLORS, order=corruption_order, ax=ax)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xticks(range(len(corruption_order)))
    ax.set_xticklabels([_wrap_label(label, width=12) for label in corruption_order])
    _legend_outside(fig, ax, title="Backbone")
    _style(ax, "Corrupted validation accuracy by corruption type",
           xlabel="Corruption", ylabel="Corrupted accuracy")
    _save(fig, os.path.join(output_dir, "s44_corrupted_accuracy_by_condition.png"))

    # --- Grouped bar: relative robustness (with reference line at 1.0) ---
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    sns.barplot(data=s44, x="Corruption", y="Relative Robustness",
                hue="Model", palette=MODEL_COLORS, order=corruption_order, ax=ax)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.3,
               label="Clean baseline (RR=1)")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(len(corruption_order)))
    ax.set_xticklabels([_wrap_label(label, width=12) for label in corruption_order])
    _legend_outside(fig, ax, title="Backbone")
    _style(ax, "Relative robustness by corruption type (RR = Acc_corr / Acc_clean)",
           xlabel="Corruption", ylabel="Relative robustness")
    _save(fig, os.path.join(output_dir, "s44_relative_robustness_by_condition.png"))

    # --- Aggregate summary: separate panels for RR and CE ---
    agg = (
        s44.groupby("Model", as_index=False)
           .agg(mean_rr=("Relative Robustness", "mean"),
                mean_ce=("Corruption Error", "mean"))
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=False)
    bar_colors = [MODEL_COLORS.get(m, PALETTE[0]) for m in agg["Model"]]

    sns.barplot(
        data=agg,
        x="Model",
        y="mean_rr",
        hue="Model",
        dodge=False,
        palette=MODEL_COLORS,
        legend=False,
        ax=ax1,
    )
    ax1.set_xticks(range(len(agg)))
    ax1.set_xticklabels([_short_model_label(label) for label in agg["Model"]])
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _style(ax1, "Mean relative robustness", xlabel="Backbone", ylabel="RR")
    for patch, value in zip(ax1.patches, agg["mean_rr"]):
        ax1.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    sns.barplot(
        data=agg,
        x="Model",
        y="mean_ce",
        hue="Model",
        dodge=False,
        palette=MODEL_COLORS,
        legend=False,
        ax=ax2,
    )
    ax2.set_xticks(range(len(agg)))
    ax2.set_xticklabels([_short_model_label(label) for label in agg["Model"]])
    ax2.set_ylim(0, 0.48)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _style(ax2, "Mean corruption error", xlabel="Backbone", ylabel="CE")
    for patch, value in zip(ax2.patches, agg["mean_ce"]):
        ax2.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("Aggregate robustness summary", fontsize=15, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "s44_aggregate_summary.png"))


# ── Scenario 4.5 — Layer-wise feature probing plots ─────────────────────────

def plot_s45(s45: pd.DataFrame, output_dir: str) -> None:
    """Generate summary plots for Scenario 4.5 (layer-wise probing)."""
    depth_to_index = {"early": 0, "middle": 1, "final": 2}
    depth_labels   = ["Early", "Middle", "Final"]

    s45 = s45.copy()
    s45["Depth_idx"] = s45["Layer"].map(depth_to_index)
    s45 = s45.sort_values(["Model", "Depth_idx"])

    # --- Line: probe accuracy vs depth ---
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.lineplot(
        data=s45,
        x="Depth_idx",
        y="Probe Test Acc (%)",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        linewidth=2.3,
        markersize=9,
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(depth_labels)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _legend_outside(fig, ax, title="Backbone")
    _style(ax, "Probe test accuracy vs network depth",
           xlabel="Layer depth", ylabel="Linear probe accuracy")
    _save(fig, os.path.join(output_dir, "s45_probe_accuracy_vs_depth.png"))

    # --- Line: feature norm vs depth ---
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.lineplot(
        data=s45,
        x="Depth_idx",
        y="Mean Feature Norm",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        linewidth=2.3,
        markersize=9,
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(depth_labels)
    _legend_outside(fig, ax, title="Backbone")
    _style(ax, "Mean feature L2 norm vs network depth",
           xlabel="Layer depth", ylabel="Mean feature norm")
    _save(fig, os.path.join(output_dir, "s45_feature_norm_vs_depth.png"))

    # --- Grouped bar: probe accuracy per model per depth ---
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=s45, x="Model", y="Probe Test Acc (%)",
                hue="Layer", palette="colorblind", ax=ax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(title="Depth", frameon=True, fontsize=10)
    _style(ax, "Probe accuracy by backbone and layer depth",
           xlabel="Backbone", ylabel="Linear probe accuracy")
    _save(fig, os.path.join(output_dir, "s45_probe_accuracy_grouped.png"))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the full consolidated plot suite from saved CSV artifacts."
    )
    parser.add_argument("--results_dir", default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--out_dir", default="results/all_plots",
                        help="Output directory for generated plots (default: results/all_plots/)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    s41 = _load(args.results_dir, "s41/summary_table.csv")
    s42 = _load(args.results_dir, "s42/strategy_comparison_table.csv")
    s43 = _load(args.results_dir, "s43/fewshot_table.csv")
    s44 = _load(args.results_dir, "s44/robustness_summary.csv")
    s45 = _load(args.results_dir, "s45/depth_probing_table.csv")

    print("Generating Scenario 4.1 summary plots …")
    plot_s41(s41, args.out_dir)

    print("Generating Scenario 4.2 summary plots …")
    plot_s42(s42, args.out_dir)

    print("Generating Scenario 4.3 summary plots …")
    plot_s43(s43, args.out_dir)

    print("Generating Scenario 4.4 summary plots …")
    plot_s44(s44, args.out_dir)

    print("Generating Scenario 4.5 summary plots …")
    plot_s45(s45, args.out_dir)

    print(f"\nDone. Plots written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
