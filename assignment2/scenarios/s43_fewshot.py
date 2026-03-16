"""
Scenario 4.3 — Few-Shot Learning Analysis

Evaluate data efficiency of each backbone under limited supervision.
Four training regimes:
  - 100% of training data
  - 50%  of training data
  - 20%  of training data
  - 5%   of training data

All subsets generated with the same random seed (config.SEED = 123) for
reproducibility. Strategy: linear probe (frozen backbone) to isolate the
effect of data size from fine-tuning strategy.

Outputs per model (under results/s43/{model_name}/):
  - accuracy_vs_fraction.png

Summary: results/s43/fewshot_table.csv
"""

import os
import json
import pandas as pd
import torch

import config
import dataset as ds
import models as mdl
import train as trn
import efficiency as eff
from utils import visualize


def run(model_names=config.MODELS, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SCENARIO 4.3 — Few-Shot Learning Analysis  (device: {device})")
    print(f"{'='*60}")

    trn.set_seed(config.SEED)

    # Val set is always the full val split (fixed)
    train_ds, val_ds = ds.get_splits(seed=config.SEED)
    train_ds_reg = None
    if config.S43_STRONG_AUG:
        train_ds_reg = ds.AIDDataset(
            train_ds.samples,
            train_ds.class_to_idx,
            transform=ds.get_transforms(train=True, strong=True),
        )
        print("Few-shot mode: low-data regularization ENABLED (<100% data)")
    val_loader = ds.make_loader(val_ds, shuffle=False)

    all_summary_rows = []
    fewshot_results_all = {}   # {model_name: {fraction: val_acc}}

    for model_name in model_names:
        display = config.MODEL_DISPLAY_NAMES[model_name]
        out_dir = os.path.join(config.RESULTS_DIR, "s43", model_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[4.3] Model: {display}")

        fewshot_results_all[model_name] = {}
        fraction_rows = []

        acc_100 = None   # will be filled by 100% run

        for fraction in config.FEWSHOT_FRACTIONS:
            label = f"{int(fraction*100)}%"
            print(f"\n  Fraction: {label}")

            # Build subset. Keep 100% regime unchanged; regularize only low-data.
            source_ds = train_ds
            if fraction < 1.0 and train_ds_reg is not None:
                source_ds = train_ds_reg
            subset = ds.get_fewshot_subset(source_ds, fraction, seed=config.SEED)
            train_loader = ds.make_loader(subset, shuffle=True)

            trn.set_seed(config.SEED)
            model = mdl.create_model(model_name)
            mdl.freeze_backbone(model, model_name)
            model = model.to(device)

            if fraction == 1.0:
                eff.report_efficiency(model.cpu(), display)
                model = model.to(device)

            is_low_data = fraction < 1.0
            epochs = config.EPOCHS_FULL if fraction == 1.0 else config.EPOCHS_FEWSHOT
            lr = config.LR_PROBE if not is_low_data else config.S43_LOW_DATA_LR
            wd = config.WEIGHT_DECAY if not is_low_data else config.S43_WEIGHT_DECAY
            optimizer, scheduler = trn.make_optimizer_scheduler(
                model, lr=lr, epochs=epochs, weight_decay=wd
            )

            history = trn.run_training(
                model, train_loader, val_loader,
                optimizer, scheduler,
                epochs=epochs,
                device=device,
                grad_norm_log=False,
                early_stop_patience=config.S43_EARLY_STOP_PATIENCE if is_low_data else 0,
                early_stop_min_delta=config.EARLY_STOP_MIN_DELTA,
            )

            best_val = history["best_val_acc"]
            best_epoch = history["best_epoch"]
            final_val = history["val_acc"][-1]
            final_train = history["train_acc"][-1]
            train_val_gap = final_train - final_val

            print(
                f"  Best Val Acc: {best_val*100:.2f}% (epoch {best_epoch})  |  "
                f"Train-Val Gap: {train_val_gap*100:.2f}%"
            )

            fewshot_results_all[model_name][fraction] = best_val

            if fraction == 1.0:
                acc_100 = best_val

            fraction_rows.append({
                "fraction": fraction,
                "label": label,
                "best_val_acc": best_val,
                "final_val_acc": final_val,
                "final_train_acc": final_train,
                "train_val_gap": train_val_gap,
                "n_train_samples": len(subset),
                "epochs_trained": len(history["val_acc"]),
            })

            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_state_dict": history["best_state_dict"],
                "history": history,
                "fraction": fraction,
                "best_val_acc": best_val,
                "best_epoch": best_epoch,
            }, os.path.join(out_dir, f"checkpoint_{label}.pt"))

        # Compute relative performance drop Δ = (Acc_100 - Acc_5) / Acc_100
        acc_5 = fewshot_results_all[model_name].get(0.05, 0.0)
        delta = (acc_100 - acc_5) / acc_100 if acc_100 and acc_100 > 0 else 0.0
        print(f"\n  Δ (relative perf drop 100% → 5%): {delta*100:.1f}%")

        # Summary rows
        for row in fraction_rows:
            all_summary_rows.append({
                "Model": display,
                "Data Fraction": row["label"],
                "N Train Samples": row["n_train_samples"],
                "Epochs Trained": row["epochs_trained"],
                "Best Val Acc (%)": round(row["best_val_acc"] * 100, 2),
                "Final Train Acc (%)": round(row["final_train_acc"] * 100, 2),
                "Train-Val Gap (%)": round(row["train_val_gap"] * 100, 2),
                "Relative Drop Δ (%)": round(delta * 100, 2) if row["fraction"] == 0.05 else "",
            })

    # ── Plots ──────────────────────────────────────────────────────────────────
    visualize.plot_fewshot_accuracy(
        fewshot_results_all,
        title="Few-Shot Learning: Validation Accuracy vs Training Data Fraction",
        save_path=os.path.join(config.RESULTS_DIR, "s43", "accuracy_vs_fraction.png"),
    )

    # ── Summary table ──────────────────────────────────────────────────────────
    summary_path = os.path.join(config.RESULTS_DIR, "s43", "fewshot_table.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df = pd.DataFrame(all_summary_rows)
    df.to_csv(summary_path, index=False)
    print(f"\n[4.3] Summary saved to {summary_path}")
    print(df.to_string(index=False))

    return all_summary_rows


if __name__ == "__main__":
    run()
