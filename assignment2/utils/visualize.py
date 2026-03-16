"""
Visualisation utilities for GNR638 Assignment 2.

All public functions accept data, a title, and a save path.
They save the figure at 300 dpi and return the Figure object for
optional notebook display.

Style contract
--------------
* We use seaborn's 'whitegrid' theme with 'talk' context so every
  figure is readable in a report without extra font-size tuning.
* The colour palette is 'colorblind' throughout for accessibility.
* Despine removes the top and right axes borders for a clean look.
* We never mutate global matplotlib/seaborn state inside a function —
  the theme is set once at module load time.
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # headless: no display needed

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config

# ── Global theme ──────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    context="talk",
    palette="colorblind",
    rc={
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
    },
)
PALETTE = sns.color_palette("colorblind")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str, dpi: int = 300) -> None:
    """Save *fig* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _style_ax(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=14, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    sns.despine(ax=ax)


def _humanize_label(label: str) -> str:
    """Turn raw IDs/CamelCase labels into readable text."""
    text = str(label).replace("_", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wrap_label(label: str, width: int = 12) -> str:
    """Wrap a label onto multiple lines for crowded axes."""
    return textwrap.fill(_humanize_label(label), width=width, break_long_words=False)


def _display_model_name(model_name: str) -> str:
    """Return a stable display label for either raw keys or already-formatted names."""
    if model_name in config.MODEL_DISPLAY_NAMES:
        return config.MODEL_DISPLAY_NAMES[model_name]
    if model_name in config.MODEL_DISPLAY_NAMES.values():
        return model_name
    return _humanize_label(model_name)


def _legend_outside(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    ncol: int = 1,
    anchor_x: float = 1.02,
) -> None:
    """Place the legend outside the plotting area for readability."""
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(anchor_x, 0.5),
        frameon=True,
        title=title,
        ncol=ncol,
    )
    if legend is not None:
        fig.subplots_adjust(right=0.80)


# ── Training / validation accuracy curves ────────────────────────────────────

def plot_accuracy_curves(
    history: Dict[str, list],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Plot training and validation accuracy over epochs.

    Parameters
    ----------
    history : dict
        Must contain ``'train_acc'`` and ``'val_acc'`` as lists of floats
        (values in [0, 1]).
    title : str
        Figure title.
    save_path : str
        Destination file path (.png).
    """
    train_pcts = [a * 100 for a in history["train_acc"]]
    val_pcts   = [a * 100 for a in history["val_acc"]]
    epochs     = list(range(1, len(train_pcts) + 1))
    df = pd.DataFrame(
        {
            "Epoch": epochs + epochs,
            "Accuracy": train_pcts + val_pcts,
            "Split": ["Train"] * len(epochs) + ["Validation"] * len(epochs),
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Accuracy",
        hue="Split",
        style="Split",
        markers=True,
        dashes=False,
        linewidth=2.2,
        markersize=7,
        ax=ax,
    )
    ax.fill_between(epochs, train_pcts, val_pcts, alpha=0.08, color=PALETTE[2])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlim(1, max(epochs))
    ax.legend(frameon=True, title="Series")
    _style_ax(ax, title, xlabel="Epoch", ylabel="Accuracy")
    _save(fig, save_path)
    return fig


def plot_loss_curves(
    history: Dict[str, list],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Plot training and validation loss over epochs.

    Parameters
    ----------
    history : dict
        Must contain ``'train_loss'`` and ``'val_loss'`` as lists of floats.
    """
    epochs     = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]

    df = pd.DataFrame(
        {
            "Epoch": epochs + epochs,
            "Loss": train_loss + val_loss,
            "Split": ["Train loss"] * len(epochs) + ["Validation loss"] * len(epochs),
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Loss",
        hue="Split",
        style="Split",
        markers=True,
        dashes=False,
        linewidth=2.2,
        markersize=7,
        ax=ax,
    )
    ax.legend(frameon=True, title="Series")
    _style_ax(ax, title, xlabel="Epoch", ylabel="Cross-entropy loss")
    _save(fig, save_path)
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Plot a row-normalised confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
        Raw count matrix (rows = true class, columns = predicted class).
    class_names : list of str
        Ordered class labels.
    """
    n_classes = len(class_names)

    # Row-normalise so every entry is a recall fraction in [0, 1].
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_normed  = np.where(row_totals > 0, cm / row_totals, 0.0)

    tick_labels = [_wrap_label(name, width=11) for name in class_names]
    fig_size = max(13.5, n_classes * 0.48)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_normed,
        ax=ax,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot=(n_classes <= 15),
        fmt=".2f",
        linewidths=0.3 if n_classes <= 15 else 0,
        square=True,
        cbar_kws={"label": "Recall fraction", "shrink": 0.84},
    )
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8.5)
    _save(fig, save_path)
    return fig


# ── t-SNE embedding visualisation ────────────────────────────────────────────

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str,
    seed: int = 123,
) -> plt.Figure:
    """
    Compute 2-D t-SNE and draw a scatter plot coloured by class.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
    labels   : np.ndarray, shape (n_samples,), integer class indices
    class_names : list of str
    seed : int
        Random state for reproducibility.
    """
    print(f"  t-SNE: fitting {features.shape[0]} samples × {features.shape[1]} dims …")
    reducer = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
    embedding = reducer.fit_transform(features)

    n_classes = len(class_names)
    legend_labels = [_wrap_label(name, width=14) for name in class_names]
    df = pd.DataFrame(
        {
            "tSNE-1": embedding[:, 0],
            "tSNE-2": embedding[:, 1],
            "Class": [legend_labels[idx] for idx in labels],
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    sns.scatterplot(
        data=df,
        x="tSNE-1",
        y="tSNE-2",
        hue="Class",
        palette=sns.color_palette("husl", n_classes),
        s=22,
        alpha=0.78,
        linewidth=0,
        legend=(n_classes <= 18),
        ax=ax,
    )
    ax.set_xlabel("t-SNE dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE dimension 2", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    if n_classes <= 20:
        _legend_outside(fig, ax, title="Class", ncol=1)
    sns.despine(ax=ax)
    _save(fig, save_path)
    return fig


# ── PCA 2-D visualisation ─────────────────────────────────────────────────────

def plot_pca_2d(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str,
    seed: int = 123,
) -> plt.Figure:
    """
    Compute 2-D PCA and draw a scatter plot coloured by class.

    Used for Scenario 4.5 — the same fixed 30×30 subset should be
    passed for every model and layer to ensure fair comparison.
    """
    reducer   = PCA(n_components=2, random_state=seed)
    embedding = reducer.fit_transform(features)
    var_pct   = reducer.explained_variance_ratio_ * 100

    n_classes = len(class_names)
    legend_labels = [_wrap_label(name, width=14) for name in class_names]
    df = pd.DataFrame(
        {
            "PC1": embedding[:, 0],
            "PC2": embedding[:, 1],
            "Class": [legend_labels[idx] for idx in labels],
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="Class",
        palette=sns.color_palette("husl", n_classes),
        s=28,
        alpha=0.82,
        linewidth=0,
        legend=(n_classes <= 18),
        ax=ax,
    )
    ax.set_xlabel(f"PC 1 ({var_pct[0]:.1f}% var.)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({var_pct[1]:.1f}% var.)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    if n_classes <= 20:
        _legend_outside(fig, ax, title="Class", ncol=1)
    sns.despine(ax=ax)
    _save(fig, save_path)
    return fig


# ── Gradient norm tracking ────────────────────────────────────────────────────

def plot_grad_norms(
    norms_per_epoch: List[Dict[str, float]],
    title: str,
    save_path: str,
    max_groups: int = 10,
) -> Optional[plt.Figure]:
    """
    Plot mean gradient norm per named parameter group across training epochs.

    Parameters
    ----------
    norms_per_epoch : list of dicts
        One dict per epoch: ``{layer_prefix: mean_norm_value}``.
    max_groups : int
        Cap the number of plotted groups (keeps the legend readable).
    """
    if not norms_per_epoch:
        return None

    all_groups = sorted({group for epoch_dict in norms_per_epoch for group in epoch_dict.keys()})
    if not all_groups:
        return None

    ranked_groups = sorted(
        all_groups,
        key=lambda group: max(epoch_dict.get(group, 0.0) for epoch_dict in norms_per_epoch),
        reverse=True,
    )[:max_groups]

    rows = []
    for epoch_idx, epoch_dict in enumerate(norms_per_epoch, start=1):
        for group_name in ranked_groups:
            rows.append(
                {
                    "Epoch": epoch_idx,
                    "MeanNorm": max(epoch_dict.get(group_name, 0.0), 1e-8),
                    "Group": _wrap_label(group_name, width=18),
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    sns.lineplot(
        data=df,
        x="Epoch",
        y="MeanNorm",
        hue="Group",
        style="Group",
        markers=True,
        dashes=False,
        linewidth=2.0,
        markersize=6,
        palette="tab10",
        ax=ax,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.LogFormatter())
    _legend_outside(fig, ax, title="Layer group", ncol=1)
    _style_ax(ax, title, xlabel="Epoch", ylabel="Mean gradient norm (log)")
    _save(fig, save_path)
    return fig


# ── Convergence comparison across strategies ──────────────────────────────────

def plot_convergence(
    loss_by_strategy: Dict[str, List[float]],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Overlay training loss curves for multiple fine-tuning strategies.

    Parameters
    ----------
    loss_by_strategy : dict
        ``{strategy_label: [loss_epoch_1, loss_epoch_2, …]}``.
    """
    rows = []
    for strategy, losses in loss_by_strategy.items():
        for epoch_idx, loss in enumerate(losses, start=1):
            rows.append(
                {
                    "Epoch": epoch_idx,
                    "Loss": loss,
                    "Strategy": _humanize_label(strategy),
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Loss",
        hue="Strategy",
        style="Strategy",
        markers=True,
        dashes=False,
        linewidth=2.2,
        markersize=7,
        ax=ax,
    )
    _legend_outside(fig, ax, title="Strategy")
    _style_ax(ax, title, xlabel="Epoch", ylabel="Training loss")
    _save(fig, save_path)
    return fig


# ── Fine-tuning: accuracy vs unfrozen backbone % ─────────────────────────────

def plot_accuracy_vs_unfrozen(
    strategy_results: List[Dict],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Plot train and validation accuracy as a function of the percentage of
    backbone parameters that are unfrozen.

    Parameters
    ----------
    strategy_results : list of dicts
        Each dict must contain:
        ``strategy``, ``unfrozen_pct``, ``train_acc``, ``val_acc``
        (accuracy values in [0, 1]).
    """
    df = pd.DataFrame(strategy_results).sort_values("unfrozen_pct")
    long_df = pd.DataFrame(
        {
            "UnfrozenPct": np.concatenate([df["unfrozen_pct"].to_numpy(), df["unfrozen_pct"].to_numpy()]),
            "Accuracy": np.concatenate([(df["train_acc"] * 100).to_numpy(), (df["val_acc"] * 100).to_numpy()]),
            "Series": ["Train"] * len(df) + ["Validation"] * len(df),
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sns.lineplot(
        data=long_df,
        x="UnfrozenPct",
        y="Accuracy",
        hue="Series",
        style="Series",
        markers=True,
        dashes=False,
        linewidth=2.2,
        markersize=8,
        ax=ax,
    )
    sns.scatterplot(
        data=long_df,
        x="UnfrozenPct",
        y="Accuracy",
        hue="Series",
        style="Series",
        legend=False,
        s=70,
        ax=ax,
    )

    for _, row in df.iterrows():
        ax.annotate(
            _wrap_label(row["strategy"], width=10),
            (row["unfrozen_pct"], row["val_acc"] * 100),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=9,
        )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(frameon=True, title="Series")
    _style_ax(ax, title,
              xlabel="Unfrozen backbone parameters (%)",
              ylabel="Accuracy")
    _save(fig, save_path)
    return fig


# ── Fine-tuning: accuracy vs tuned parameter count ───────────────────────────

def plot_performance_vs_tuned_params(
    strategy_results: List[Dict],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Scatter plot of best validation accuracy against number of trainable
    parameters (log-scaled x-axis).

    Parameters
    ----------
    strategy_results : list of dicts
        Each dict must contain: ``strategy``, ``trainable_params``,
        ``best_val_acc`` (in [0, 1]).
    """
    df = pd.DataFrame(strategy_results)
    if df.empty:
        return None

    trainable_millions = df["trainable_params"].astype(float) / 1e6
    best_val_pcts      = df["best_val_acc"].astype(float) * 100.0

    df_plot = pd.DataFrame(
        {
            "TrainableM": trainable_millions,
            "BestVal": best_val_pcts,
            "Strategy": [_humanize_label(s) for s in df["strategy"]],
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    sns.scatterplot(
        data=df_plot,
        x="TrainableM",
        y="BestVal",
        hue="Strategy",
        style="Strategy",
        s=110,
        ax=ax,
    )
    for _, row in df.iterrows():
        ax.annotate(
            _wrap_label(row["strategy"], width=10),
            (row["trainable_params"] / 1e6, row["best_val_acc"] * 100.0),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _legend_outside(fig, ax, title="Strategy")
    _style_ax(ax, title,
              xlabel="Trainable parameters (millions, log scale)",
              ylabel="Best validation accuracy")
    _save(fig, save_path)
    return fig


# ── Few-shot accuracy vs data fraction ───────────────────────────────────────

def plot_fewshot_accuracy(
    results: Dict[str, Dict[float, float]],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Line plot of validation accuracy vs training data fraction for multiple
    models.

    Parameters
    ----------
    results : dict
        ``{model_name: {fraction_float: val_acc_float}}``.
        Fractions are e.g. 0.05, 0.20, 1.0.
    """
    rows = []
    for model_name, fraction_to_acc in results.items():
        display_name = _display_model_name(model_name)
        for fraction, acc in sorted(fraction_to_acc.items()):
            rows.append(
                {
                    "FractionPct": fraction * 100,
                    "Accuracy": acc * 100,
                    "Model": display_name,
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sns.lineplot(
        data=df,
        x="FractionPct",
        y="Accuracy",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        linewidth=2.3,
        markersize=8,
        ax=ax,
    )

    x_ticks = sorted(df["FractionPct"].unique())
    ax.set_xticks(x_ticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _legend_outside(fig, ax, title="Backbone")
    _style_ax(ax, title,
              xlabel="Training data used (%)",
              ylabel="Validation accuracy")
    _save(fig, save_path)
    return fig


# ── Corruption robustness bar chart ──────────────────────────────────────────

def plot_robustness_bars(
    robustness: Dict[str, Dict[str, float]],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Grouped bar chart of relative robustness for each corruption type.

    Parameters
    ----------
    robustness : dict
        ``{model_name: {corruption_label: relative_robustness_float}}``.
        Relative robustness = Acc_corrupted / Acc_clean.
    """
    rows = []
    for model_name, corruption_map in robustness.items():
        for corruption_label, value in corruption_map.items():
            rows.append(
                {
                    "Model": model_name,
                    "Corruption": _wrap_label(corruption_label, width=14),
                    "RelativeRobustness": value,
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13.2, 6.1))
    sns.barplot(
        data=df,
        x="Corruption",
        y="RelativeRobustness",
        hue="Model",
        ax=ax,
    )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2, label="Clean baseline")
    ax.tick_params(axis="x", rotation=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=2, frameon=True, title="Backbone")
    fig.subplots_adjust(bottom=0.27)
    _style_ax(ax, title, ylabel="Relative robustness (Acc_corrupted / Acc_clean)")
    _save(fig, save_path)
    return fig


# ── Layer-wise probe accuracy vs depth ───────────────────────────────────────

def plot_accuracy_vs_depth(
    results: Dict[str, Dict[str, float]],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Line plot of linear probe accuracy at early, middle, and final network
    depth for multiple models.

    Parameters
    ----------
    results : dict
        ``{model_name: {depth_label: probe_acc_float}}``.
        Depth labels should be: ``'early'``, ``'middle'``, ``'final'``.
    """
    depth_order = ["early", "middle", "final"]
    rows = []
    for model_name, depth_to_acc in results.items():
        display_name = _display_model_name(model_name)
        for depth in depth_order:
            if depth in depth_to_acc:
                rows.append(
                    {
                        "Depth": depth.capitalize(),
                        "Accuracy": depth_to_acc[depth] * 100,
                        "Model": display_name,
                    }
                )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    sns.lineplot(
        data=df,
        x="Depth",
        y="Accuracy",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        linewidth=2.3,
        markersize=9,
        sort=False,
        ax=ax,
    )

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _legend_outside(fig, ax, title="Backbone")
    _style_ax(ax, title,
              xlabel="Network depth",
              ylabel="Linear probe accuracy")
    _save(fig, save_path)
    return fig


# ── Feature norm bar chart ────────────────────────────────────────────────────

def plot_feature_norms(
    norms: Dict[str, float],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Bar chart of mean feature L2 norms across layer depth labels.

    Parameters
    ----------
    norms : dict
        ``{depth_label: mean_norm_float}``.
    """
    depth_labels = list(norms.keys())
    norm_values = list(norms.values())
    df = pd.DataFrame(
        {
            "Depth": [label.capitalize() for label in depth_labels],
            "Norm": norm_values,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    sns.barplot(data=df, x="Depth", y="Norm", hue="Depth", dodge=False, legend=False, ax=ax)
    for patch, value in zip(ax.patches, norm_values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.01 * max(norm_values),
            f"{value:.2f}",
            ha="center", va="bottom", fontsize=10,
        )
    _style_ax(ax, title,
              xlabel="Layer depth",
              ylabel="Mean feature L2 norm")
    _save(fig, save_path)
    return fig


def plot_multi_strategy_accuracy_curves(
    strategy_histories: Dict[str, Dict[str, list]],
    title: str,
    save_path: str,
) -> plt.Figure:
    """
    Plot train/validation accuracy curves for multiple strategies on two panels.

    Parameters
    ----------
    strategy_histories : dict
        ``{strategy_name: history_dict}``, where each history contains
        ``train_acc`` and ``val_acc`` lists in [0, 1].
    """
    rows = []
    for strategy, history in strategy_histories.items():
        label = _humanize_label(strategy)
        for epoch_idx, value in enumerate(history["train_acc"], start=1):
            rows.append(
                {
                    "Epoch": epoch_idx,
                    "Accuracy": value * 100,
                    "Strategy": label,
                    "Split": "Train",
                }
            )
        for epoch_idx, value in enumerate(history["val_acc"], start=1):
            rows.append(
                {
                    "Epoch": epoch_idx,
                    "Accuracy": value * 100,
                    "Strategy": label,
                    "Split": "Validation",
                }
            )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), sharey=True)
    for ax, split in zip(axes, ["Train", "Validation"]):
        split_df = df[df["Split"] == split]
        sns.lineplot(
            data=split_df,
            x="Epoch",
            y="Accuracy",
            hue="Strategy",
            style="Strategy",
            markers=True,
            dashes=False,
            linewidth=2.1,
            markersize=6,
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        _style_ax(ax, f"{title} -- {split}", xlabel="Epoch", ylabel="Accuracy" if split == "Train" else "")
        if split == "Train":
            _legend_outside(fig, ax, title="Strategy")
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    _save(fig, save_path)
    return fig
