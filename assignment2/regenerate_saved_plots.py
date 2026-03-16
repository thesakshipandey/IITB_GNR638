#!/usr/bin/env python3
"""
Regenerate Assignment-2 plots from saved checkpoints and CSV artifacts.

This script does not retrain the expensive scenarios. It rebuilds:
  - Scenario 4.1 plots from saved linear-probe checkpoints
  - Scenario 4.2 plots from saved strategy checkpoints
  - Scenario 4.3 plot from the saved few-shot CSV
  - Scenario 4.4 plots by rerunning the lightweight evaluation-only script
  - Scenario 4.5 plots by rerunning the lightweight probing script
"""

from __future__ import annotations

import os

import pandas as pd
import torch

import config
import dataset as ds
import evaluate as ev
import models as mdl
import train as trn
from scenarios import s44_corruption, s45_layer_probing
from utils import hooks, visualize


STRATEGY_ORDER = ["linear_probe", "last_block", "selective", "full"]


def _load_checkpoint(path: str, device: torch.device) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location=device)


def regenerate_s41(device: torch.device) -> None:
    print("Regenerating Scenario 4.1 plots from saved checkpoints ...")
    trn.set_seed(config.SEED)
    train_ds, val_ds = ds.get_splits(seed=config.SEED)
    val_loader = ds.make_loader(val_ds, shuffle=False)
    class_names = train_ds.classes

    for model_name in config.MODELS:
        display = config.MODEL_DISPLAY_NAMES[model_name]
        out_dir = os.path.join(config.RESULTS_DIR, "s41", model_name)
        ckpt = _load_checkpoint(os.path.join(out_dir, "checkpoint.pt"), device=device)
        history = ckpt["history"]

        visualize.plot_accuracy_curves(
            history,
            title=f"{display} — Linear Probe: Accuracy",
            save_path=os.path.join(out_dir, "accuracy_curves.png"),
        )
        visualize.plot_loss_curves(
            history,
            title=f"{display} — Linear Probe: Loss",
            save_path=os.path.join(out_dir, "loss_curves.png"),
        )

        model = mdl.create_model(model_name)
        best_state = ckpt.get("best_state_dict") or ckpt["model_state_dict"]
        model.load_state_dict(best_state)
        model = model.to(device)

        cm = ev.compute_confusion_matrix(model, val_loader, config.NUM_CLASSES, device)
        visualize.plot_confusion_matrix(
            cm,
            class_names,
            title=f"{display} — Linear Probe: Confusion Matrix",
            save_path=os.path.join(out_dir, "confusion_matrix.png"),
        )

        extractor = hooks.FeatureExtractor(model, {"final": config.LAYER_HOOKS[model_name]["final"]})
        feat_result = extractor.extract(val_loader, device)
        extractor.remove_hooks()
        visualize.plot_tsne(
            feat_result["final"],
            feat_result["__labels__"],
            class_names,
            title=f"{display} — Linear Probe: t-SNE Feature Embeddings",
            save_path=os.path.join(out_dir, "tsne_embeddings.png"),
            seed=config.SEED,
        )


def regenerate_s42(device: torch.device) -> None:
    print("Regenerating Scenario 4.2 plots from saved checkpoints ...")
    for model_name in config.MODELS:
        display = config.MODEL_DISPLAY_NAMES[model_name]
        out_dir = os.path.join(config.RESULTS_DIR, "s42", model_name)

        strategy_histories = {}
        strategy_results = []

        for strategy_name in STRATEGY_ORDER:
            ckpt = _load_checkpoint(
                os.path.join(out_dir, f"checkpoint_{strategy_name}.pt"),
                device=device,
            )
            history = ckpt["history"]
            strategy_histories[strategy_name] = history

            if history.get("grad_norms") and any(history["grad_norms"]):
                visualize.plot_grad_norms(
                    history["grad_norms"],
                    title=f"{display} — {strategy_name}: Gradient Norms per Layer",
                    save_path=os.path.join(out_dir, f"grad_norms_{strategy_name}.png"),
                )

            strategy_results.append(
                {
                    "strategy": strategy_name,
                    "unfrozen_pct": ckpt["unfrozen_pct"],
                    "train_acc": history["train_acc"][-1],
                    "val_acc": history["val_acc"][-1],
                    "best_val_acc": ckpt["best_val_acc"],
                    "trainable_params": ckpt["trainable_params"],
                }
            )

        loss_histories = {s: strategy_histories[s]["train_loss"] for s in STRATEGY_ORDER}
        visualize.plot_convergence(
            loss_histories,
            title=f"{display} — Training Loss: Strategy Comparison",
            save_path=os.path.join(out_dir, "convergence_loss.png"),
        )
        visualize.plot_accuracy_vs_unfrozen(
            strategy_results,
            title=f"{display} — Accuracy vs Unfrozen Backbone Parameters",
            save_path=os.path.join(out_dir, "accuracy_vs_unfrozen.png"),
        )
        visualize.plot_performance_vs_tuned_params(
            strategy_results,
            title=f"{display} — Best Accuracy vs Tuned Parameters",
            save_path=os.path.join(out_dir, "accuracy_vs_tuned_params.png"),
        )
        visualize.plot_multi_strategy_accuracy_curves(
            strategy_histories,
            title=f"{display} — Strategy Accuracy Curves",
            save_path=os.path.join(out_dir, "accuracy_curves_all_strategies.png"),
        )


def regenerate_s43() -> None:
    print("Regenerating Scenario 4.3 plot from saved CSV ...")
    path = os.path.join(config.RESULTS_DIR, "s43", "fewshot_table.csv")
    df = pd.read_csv(path)

    results = {}
    fraction_map = {"100%": 1.0, "50%": 0.50, "20%": 0.20, "5%": 0.05}
    for model_name, group in df.groupby("Model"):
        results[model_name] = {}
        for _, row in group.iterrows():
            fraction = fraction_map[str(row["Data Fraction"])]
            results[model_name][fraction] = float(row["Best Val Acc (%)"]) / 100.0

    visualize.plot_fewshot_accuracy(
        results,
        title="Few-Shot Learning: Validation Accuracy vs Training Data Fraction",
        save_path=os.path.join(config.RESULTS_DIR, "s43", "accuracy_vs_fraction.png"),
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    regenerate_s41(device)
    regenerate_s42(device)
    regenerate_s43()

    print("Regenerating Scenario 4.4 plots ...")
    s44_corruption.run(model_names=config.MODELS, device=device)

    print("Regenerating Scenario 4.5 plots ...")
    s45_layer_probing.run(model_names=config.MODELS, device=device)

    print("Done. Saved plots refreshed in results/.")


if __name__ == "__main__":
    main()
