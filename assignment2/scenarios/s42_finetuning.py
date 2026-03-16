"""
Scenario 4.2 — Fine-Tuning Strategies.
"""

import json
import os

import matplotlib
import pandas as pd
import torch

import config
import dataset as ds
import efficiency as eff
import models as mdl
import train as trn
from utils import visualize

matplotlib.use("Agg")
def _unfrozen_pct(model, model_name):
    return mdl.trainable_fraction(model, model_name) * 100.0


def _make_shared_optimizer(model, model_name, epochs):
    """
    Shared optimizer policy for all strategies:
    - head LR fixed
    - backbone LR fixed
    """
    groups = mdl.get_trainable_param_groups(model, model_name)
    param_groups = []
    if groups["backbone"]:
        param_groups.append({"params": groups["backbone"], "lr": config.S42_BACKBONE_LR})
    if groups["head"]:
        param_groups.append({"params": groups["head"], "lr": config.S42_HEAD_LR})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


def run(model_names=config.MODELS, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SCENARIO 4.2 — Fine-Tuning Strategies  (device: {device})")
    print(f"{'='*60}")
    print(
        f"Shared optimizer policy: head LR={config.S42_HEAD_LR:g}, "
        f"backbone LR={config.S42_BACKBONE_LR:g} (fixed for all strategies)"
    )

    trn.set_seed(config.SEED)
    train_ds, val_ds = ds.get_splits(seed=config.SEED)
    train_loader = ds.make_loader(train_ds, shuffle=True)
    val_loader = ds.make_loader(val_ds, shuffle=False)

    all_summary_rows = []
    strategy_order = ["linear_probe", "last_block", "selective", "full"]

    for model_name in model_names:
        display = config.MODEL_DISPLAY_NAMES[model_name]
        out_dir = os.path.join(config.RESULTS_DIR, "s42", model_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[4.2] Model: {display}")

        strategy_histories = {}
        strategy_results = []

        for strategy_name in strategy_order:
            print(f"\n  Strategy: {strategy_name}")
            trn.set_seed(config.SEED)

            model = mdl.create_model(model_name)
            model = model.to(device)
            selective_info = None

            if strategy_name == "linear_probe":
                mdl.freeze_backbone(model, model_name)
            elif strategy_name == "last_block":
                mdl.freeze_backbone(model, model_name)
                mdl.unfreeze_last_block(model, model_name)
            elif strategy_name == "selective":
                mdl.freeze_backbone(model, model_name)
                selective_info = mdl.unfreeze_selective(
                    model=model,
                    model_name=model_name,
                    target_fraction=config.SELECTIVE_UNFREEZE_FRACTION,
                    calibration_loader=train_loader,
                    device=device,
                    max_calibration_batches=config.S42_SELECTIVE_CALIBRATION_BATCHES,
                )
            elif strategy_name == "full":
                mdl.unfreeze_full(model)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            if strategy_name == "linear_probe":
                eff.report_efficiency(model.cpu(), display)
                model = model.to(device)

            # Keep data-order randomness aligned across strategies.
            trn.set_seed(config.SEED)

            pct = _unfrozen_pct(model, model_name)
            counts = mdl.get_param_counts(model)
            print(f"  Trainable: {counts['trainable']:,} ({pct:.1f}% backbone unfrozen)")

            optimizer, scheduler = _make_shared_optimizer(
                model=model,
                model_name=model_name,
                epochs=config.EPOCHS_FULL,
            )

            history = trn.run_training(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                epochs=config.EPOCHS_FULL,
                device=device,
                grad_norm_log=True,
            )

            best_val = history["best_val_acc"]
            best_epoch = history["best_epoch"]
            final_val = history["val_acc"][-1]
            final_train = history["train_acc"][-1]
            print(f"  Best Val Acc: {best_val*100:.2f}% (epoch {best_epoch})")

            strategy_histories[strategy_name] = history

            # Save best-epoch checkpoint for each strategy.
            model.load_state_dict(history["best_state_dict"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_state_dict": history["best_state_dict"],
                    "history": history,
                    "strategy": strategy_name,
                    "unfrozen_pct": pct,
                    "trainable_params": counts["trainable"],
                    "best_val_acc": best_val,
                    "best_epoch": best_epoch,
                    "selective_info": selective_info,
                },
                os.path.join(out_dir, f"checkpoint_{strategy_name}.pt"),
            )

            if selective_info is not None:
                with open(os.path.join(out_dir, "selective_policy.json"), "w") as f:
                    json.dump(selective_info, f, indent=2)

            if history["grad_norms"] and any(history["grad_norms"]):
                visualize.plot_grad_norms(
                    history["grad_norms"],
                    title=f"{display} — {strategy_name}: Gradient Norms per Layer",
                    save_path=os.path.join(out_dir, f"grad_norms_{strategy_name}.png"),
                )

            best_acc_per_mparam = (
                (best_val * 100.0) / (counts["trainable"] / 1e6)
                if counts["trainable"] > 0
                else 0.0
            )

            strategy_results.append(
                {
                    "strategy": strategy_name,
                    "unfrozen_pct": pct,
                    "train_acc": final_train,
                    "val_acc": final_val,
                    "best_val_acc": best_val,
                    "trainable_params": counts["trainable"],
                    "best_acc_per_mparam": best_acc_per_mparam,
                }
            )

            all_summary_rows.append(
                {
                    "Model": display,
                    "Strategy": strategy_name,
                    "Unfrozen (%)": round(pct, 1),
                    "Trainable Params": counts["trainable"],
                    "Best Val Acc (%)": round(best_val * 100, 2),
                    "Final Val Acc (%)": round(final_val * 100, 2),
                    "Final Train Acc (%)": round(final_train * 100, 2),
                    "Train-Val Gap (%)": round((final_train - final_val) * 100, 2),
                    "Best Acc per 1M Tuned Params": round(best_acc_per_mparam, 3),
                }
            )

        loss_histories = {s: strategy_histories[s]["train_loss"] for s in strategy_order}
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

    summary_path = os.path.join(config.RESULTS_DIR, "s42", "strategy_comparison_table.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df = pd.DataFrame(all_summary_rows)
    df.to_csv(summary_path, index=False)
    print(f"\n[4.2] Summary saved to {summary_path}")
    print(df.to_string(index=False))

    return all_summary_rows


if __name__ == "__main__":
    run()
