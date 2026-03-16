"""
Scenario 4.4 — Corruption Robustness Evaluation

Evaluate models trained in Scenario 4.1 (linear probe) under controlled
distribution shifts applied ONLY at evaluation time:

  - Gaussian noise: σ ∈ {0.05, 0.1, 0.2}
  - Motion blur:    kernel_size = 15
  - Brightness shift: factor = 1.5

Metrics:
  - Validation accuracy under each corruption
  - Corruption Error = 1 - Acc_corrupted
  - Relative Robustness = Acc_corrupted / Acc_clean

Outputs per model (under results/s44/{model_name}/):
  - robustness_table.csv

Summary / visualisation:
  - results/s44/corruption_comparison.png
  - results/s44/robustness_summary.csv
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config
import dataset as ds
import models as mdl
import train as trn
import evaluate as ev
import efficiency as eff
from utils import corruption as corr, visualize


def _eval_with_corruption(model, val_samples, class_to_idx, transform,
                           device, batch_size=config.BATCH_SIZE):
    """Build a corrupted DataLoader and return accuracy."""
    corrupted_ds = ds.AIDDataset(val_samples, class_to_idx, transform=transform)
    loader = DataLoader(corrupted_ds, batch_size=batch_size, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    metrics = ev.compute_all_metrics(model, loader, config.NUM_CLASSES, device)
    return metrics["top1_acc"]


def run(model_names=config.MODELS, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SCENARIO 4.4 — Corruption Robustness Evaluation  (device: {device})")
    print(f"{'='*60}")

    trn.set_seed(config.SEED)

    # ── Load val data (underlying samples, not transformed yet) ───────────────
    _, val_ds = ds.get_splits(seed=config.SEED)
    val_samples    = val_ds.samples
    class_to_idx   = val_ds.class_to_idx

    # ── Define corruption configurations ──────────────────────────────────────
    corruption_configs = []
    for sigma in config.CORRUPTION_SIGMAS:
        corruption_configs.append({
            "name":  "gaussian_noise",
            "label": f"Gaussian σ={sigma}",
            "level": sigma,
        })
    corruption_configs.append({
        "name":  "motion_blur",
        "label": f"MotionBlur k={config.MOTION_BLUR_KERNEL}",
        "level": config.MOTION_BLUR_KERNEL,
    })
    corruption_configs.append({
        "name":  "brightness_shift",
        "label": f"Brightness f={config.BRIGHTNESS_FACTOR}",
        "level": config.BRIGHTNESS_FACTOR,
    })

    all_summary_rows = []
    robustness_all = {}   # {model_name: {label: relative_robustness}}

    for model_name in model_names:
        display = config.MODEL_DISPLAY_NAMES[model_name]
        out_dir = os.path.join(config.RESULTS_DIR, "s44", model_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[4.4] Model: {display}")

        # ── Load checkpoint from Scenario 4.1 ────────────────────────────────
        ckpt_path = os.path.join(config.RESULTS_DIR, "s41", model_name, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found at {ckpt_path}. Training from scratch...")
            trn.set_seed(config.SEED)
            model = mdl.create_model(model_name)
            mdl.freeze_backbone(model, model_name)
            model = model.to(device)
            train_ds, _ = ds.get_splits(seed=config.SEED)
            train_loader = ds.make_loader(train_ds, shuffle=True)
            clean_val_loader = ds.make_loader(val_ds, shuffle=False)
            optimizer, scheduler = trn.make_optimizer_scheduler(
                model, lr=config.LR_PROBE, epochs=config.EPOCHS_FULL
            )
            trn.run_training(model, train_loader, clean_val_loader, optimizer, scheduler,
                              epochs=config.EPOCHS_FULL, device=device)
        else:
            model = mdl.create_model(model_name)
            ckpt = torch.load(ckpt_path, map_location=device)
            best_state = ckpt.get("best_state_dict")
            model.load_state_dict(best_state if best_state is not None else ckpt["model_state_dict"])

        model = model.to(device)
        model.eval()

        eff.report_efficiency(model.cpu(), display)
        model = model.to(device)

        # ── Clean accuracy (baseline) ─────────────────────────────────────────
        clean_transform = ds.get_transforms(train=False)
        acc_clean = _eval_with_corruption(model, val_samples, class_to_idx,
                                           clean_transform, device)
        print(f"  Clean Val Acc: {acc_clean*100:.2f}%")

        robustness_all[display] = {}
        model_rows = []

        for cfg in corruption_configs:
            c_transform = corr.get_corrupted_transforms(
                cfg["name"], level=cfg["level"], base_size=config.IMAGE_SIZE
            )
            acc_corr = _eval_with_corruption(model, val_samples, class_to_idx,
                                              c_transform, device)
            corr_error   = 1.0 - acc_corr
            rel_robust   = acc_corr / acc_clean if acc_clean > 0 else 0.0

            print(f"  {cfg['label']:30s}  Acc: {acc_corr*100:.2f}%  "
                  f"CE: {corr_error:.4f}  RR: {rel_robust:.4f}")

            robustness_all[display][cfg["label"]] = rel_robust

            model_rows.append({
                "Model": display,
                "Corruption": cfg["label"],
                "Clean Acc (%)": round(acc_clean * 100, 2),
                "Corrupted Acc (%)": round(acc_corr * 100, 2),
                "Corruption Error": round(corr_error, 4),
                "Relative Robustness": round(rel_robust, 4),
            })

        # Per-model CSV
        model_df = pd.DataFrame(model_rows)
        model_df.to_csv(os.path.join(out_dir, "robustness_table.csv"), index=False)
        all_summary_rows.extend(model_rows)

    # ── Plots ──────────────────────────────────────────────────────────────────
    visualize.plot_robustness_bars(
        robustness_all,
        title="Corruption Robustness: Relative Robustness Across Models",
        save_path=os.path.join(config.RESULTS_DIR, "s44", "corruption_comparison.png"),
    )

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = os.path.join(config.RESULTS_DIR, "s44", "robustness_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df = pd.DataFrame(all_summary_rows)
    df.to_csv(summary_path, index=False)
    print(f"\n[4.4] Summary saved to {summary_path}")
    print(df.to_string(index=False))

    return all_summary_rows


if __name__ == "__main__":
    run()
