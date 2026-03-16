"""
Scenario 4.1 — Linear Probe Transfer

Freeze all backbone parameters; train only the linear classification head.
Study transferability of ImageNet pre-trained representations to the AID dataset.

Outputs per model (under results/s41/{model_name}/):
  - accuracy_curves.png
  - loss_curves.png
  - confusion_matrix.png
  - tsne_embeddings.png
  - metrics.json

Summary table: results/s41/summary_table.csv
"""

import os
import json
import pandas as pd
import torch

import config
import dataset as ds
import models as mdl
import train as trn
import evaluate as ev
import efficiency as eff
from utils import hooks, visualize


def run(model_names=config.MODELS, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SCENARIO 4.1 — Linear Probe Transfer  (device: {device})")
    print(f"{'='*60}")

    trn.set_seed(config.SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = ds.get_splits(seed=config.SEED)
    train_loader = ds.make_loader(train_ds, shuffle=True)
    val_loader   = ds.make_loader(val_ds,   shuffle=False)
    class_names  = train_ds.classes

    summary_rows = []

    for model_name in model_names:
        display = config.MODEL_DISPLAY_NAMES[model_name]
        out_dir = os.path.join(config.RESULTS_DIR, "s41", model_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[4.1] Model: {display}")

        # ── Build model ───────────────────────────────────────────────────────
        trn.set_seed(config.SEED)  # ensure identical head init
        model = mdl.create_model(model_name)
        mdl.freeze_backbone(model, model_name)
        model = model.to(device)

        # ── Efficiency metrics ────────────────────────────────────────────────
        eff_info = eff.report_efficiency(model.cpu(), display)
        model = model.to(device)

        param_counts = mdl.get_param_counts(model)
        print(f"  Trainable params: {param_counts['trainable']:,} / {param_counts['total']:,}")

        # ── Train ─────────────────────────────────────────────────────────────
        optimizer, scheduler = trn.make_optimizer_scheduler(
            model, lr=config.LR_PROBE, epochs=config.EPOCHS_FULL
        )
        print(f"  Training linear head for {config.EPOCHS_FULL} epochs...")
        history = trn.run_training(
            model, train_loader, val_loader,
            optimizer, scheduler,
            epochs=config.EPOCHS_FULL,
            device=device,
            grad_norm_log=False,
        )

        best_val_acc = history["best_val_acc"]
        best_epoch = history["best_epoch"]
        final_val_acc = history["val_acc"][-1]
        final_train_acc = history["train_acc"][-1]
        print(
            f"  Best Val Acc: {best_val_acc*100:.2f}% (epoch {best_epoch})  |  "
            f"Final Val Acc: {final_val_acc*100:.2f}%"
        )

        # Use best-epoch weights for evaluation plots and downstream robustness.
        model.load_state_dict(history["best_state_dict"])

        # ── Save model checkpoint ─────────────────────────────────────────────
        ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_state_dict": history["best_state_dict"],
            "history": history,
            "model_name": model_name,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        }, ckpt_path)

        # ── Plots ─────────────────────────────────────────────────────────────
        visualize.plot_accuracy_curves(
            history, title=f"{display} — Linear Probe: Accuracy",
            save_path=os.path.join(out_dir, "accuracy_curves.png")
        )
        visualize.plot_loss_curves(
            history, title=f"{display} — Linear Probe: Loss",
            save_path=os.path.join(out_dir, "loss_curves.png")
        )

        # Confusion matrix
        cm = ev.compute_confusion_matrix(model, val_loader, config.NUM_CLASSES, device)
        visualize.plot_confusion_matrix(
            cm, class_names,
            title=f"{display} — Linear Probe: Confusion Matrix",
            save_path=os.path.join(out_dir, "confusion_matrix.png")
        )

        # ── t-SNE embedding visualisation ─────────────────────────────────────
        print(f"  Extracting features for t-SNE...")
        hook_layer = config.LAYER_HOOKS[model_name]["final"]
        extractor = hooks.FeatureExtractor(model, {"final": hook_layer})
        feat_result = extractor.extract(val_loader, device)
        extractor.remove_hooks()

        features = feat_result["final"]          # (N, D)
        feat_labels = feat_result["__labels__"]  # (N,)

        visualize.plot_tsne(
            features, feat_labels, class_names,
            title=f"{display} — Linear Probe: t-SNE Feature Embeddings",
            save_path=os.path.join(out_dir, "tsne_embeddings.png"),
            seed=config.SEED,
        )

        # ── Save metrics ──────────────────────────────────────────────────────
        metrics = {
            "model": model_name,
            "display_name": display,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "final_val_acc": final_val_acc,
            "final_train_acc": final_train_acc,
            "params_total": eff_info["params"],
            "macs": eff_info["macs"],
            "flops": eff_info["flops"],
            "trainable_params": param_counts["trainable"],
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        summary_rows.append({
            "Model": display,
            "Best Val Acc (%)": round(best_val_acc * 100, 2),
            "Final Val Acc (%)": round(final_val_acc * 100, 2),
            "Final Train Acc (%)": round(final_train_acc * 100, 2),
            "Train-Val Gap (%)": round((final_train_acc - final_val_acc) * 100, 2),
            "Total Params": eff_info["params"],
            "Trainable Params": param_counts["trainable"],
            "MACs": eff_info["macs"],
            "FLOPs": eff_info["flops"],
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_path = os.path.join(config.RESULTS_DIR, "s41", "summary_table.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df.to_csv(summary_path, index=False)
    print(f"\n[4.1] Summary saved to {summary_path}")
    print(df.to_string(index=False))

    return summary_rows


if __name__ == "__main__":
    run()
