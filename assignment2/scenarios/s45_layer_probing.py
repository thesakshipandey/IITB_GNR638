"""
Scenario 4.5 — Layer-Wise Feature Probing

Extract intermediate representations from early, middle, and final layers of
each frozen pretrained backbone. Train separate sklearn LogisticRegression
classifiers on these features to probe semantic abstraction at each depth.

Fixed PCA subset: 30 classes × 30 samples = 900 images, same indices across
ALL models and layers.

Layer selections:
  ResNet50:       layer1 (early), layer2 (middle), layer4 (final)
  EfficientNet-B0: blocks.0 (early), blocks.3 (middle), blocks.6 (final)
  ConvNeXt-Tiny:  stages.0 (early), stages.1 (middle), stages.3 (final)

Outputs per model (under results/s45/{model_name}/):
  - accuracy_vs_depth.png
  - feature_norms.png
  - pca_2d_early.png, pca_2d_middle.png, pca_2d_final.png

Summary: results/s45/depth_probing_table.csv
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import config
import dataset as ds
import models as mdl
import train as trn
import efficiency as eff
from utils import hooks, visualize


def _probe_layer(features: np.ndarray, labels: np.ndarray, seed: int = config.SEED):
    """
    Train a logistic regression on 80% of features, test on 20%.
    Returns (train_acc, test_acc, mean_norm).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=seed
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    pipe.fit(X_tr, y_tr)
    train_acc = pipe.score(X_tr, y_tr)
    test_acc  = pipe.score(X_te, y_te)
    return train_acc, test_acc


def run(model_names=config.MODELS, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SCENARIO 4.5 — Layer-Wise Feature Probing  (device: {device})")
    print(f"{'='*60}")

    trn.set_seed(config.SEED)

    _, val_ds = ds.get_splits(seed=config.SEED)
    val_loader = ds.make_loader(val_ds, shuffle=False)

    # Fixed PCA subset — created once, reused for all models/layers
    pca_subset, pca_labels = ds.get_fixed_pca_subset(val_ds,
                                                       n_per_class=config.LAYER_PROBE_SUBSET_PER_CLASS,
                                                       seed=config.SEED)
    pca_loader = DataLoader(pca_subset, batch_size=64, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)
    pca_labels_arr = np.array(pca_labels)

    all_summary_rows = []
    depth_results_all = {}   # {model_name: {layer_label: test_acc}}

    for model_name in model_names:
        display = config.MODEL_DISPLAY_NAMES[model_name]
        out_dir = os.path.join(config.RESULTS_DIR, "s45", model_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[4.5] Model: {display}")

        # Build frozen pretrained model
        trn.set_seed(config.SEED)
        model = mdl.create_model(model_name, pretrained=True)
        mdl.freeze_backbone(model, model_name)
        model = model.to(device)

        eff.report_efficiency(model.cpu(), display)
        model = model.to(device)

        # Layer map for this model
        layer_map = config.LAYER_HOOKS[model_name]   # {"early": path, "middle": path, "final": path}
        print(f"  Probing layers: {layer_map}")

        # ── Extract features for all layers at once ────────────────────────────
        extractor = hooks.FeatureExtractor(model, layer_map)

        # Full val set features (for logistic regression probing)
        print("  Extracting features for val set...")
        val_features = extractor.extract(val_loader, device)
        val_labels = val_features["__labels__"]

        # PCA subset features (same model, same layers)
        print("  Extracting features for PCA subset...")
        pca_features = extractor.extract(pca_loader, device)

        extractor.remove_hooks()

        # ── Probe each layer ───────────────────────────────────────────────────
        depth_results_all[model_name] = {}
        norms = {}

        for layer_label, layer_path in layer_map.items():
            feat = val_features[layer_label]    # (N, D)
            train_acc, test_acc = _probe_layer(feat, val_labels)
            mean_norm = float(np.linalg.norm(feat, axis=1).mean())
            norms[layer_label] = mean_norm

            print(f"  [{layer_label:8s}] path={layer_path:20s}  "
                  f"Probe Train Acc: {train_acc*100:.2f}%  "
                  f"Test Acc: {test_acc*100:.2f}%  "
                  f"Mean Norm: {mean_norm:.4f}")

            depth_results_all[model_name][layer_label] = test_acc

            # PCA 2D plot for this layer (fixed subset)
            pca_feat = pca_features[layer_label]   # (900, D)
            class_names = val_ds.classes
            visualize.plot_pca_2d(
                pca_feat, pca_labels_arr, class_names,
                title=f"{display} — PCA 2D ({layer_label} layer: {layer_path})",
                save_path=os.path.join(out_dir, f"pca_2d_{layer_label}.png"),
                seed=config.SEED,
            )

            all_summary_rows.append({
                "Model": display,
                "Layer": layer_label,
                "Layer Path": layer_path,
                "Probe Train Acc (%)": round(train_acc * 100, 2),
                "Probe Test Acc (%)": round(test_acc * 100, 2),
                "Mean Feature Norm": round(mean_norm, 4),
                "Feature Dim": feat.shape[1],
            })

        # ── Per-model plots ────────────────────────────────────────────────────
        visualize.plot_accuracy_vs_depth(
            {model_name: depth_results_all[model_name]},
            title=f"{display} — Linear Probe Accuracy vs Layer Depth",
            save_path=os.path.join(out_dir, "accuracy_vs_depth.png"),
        )
        visualize.plot_feature_norms(
            norms,
            title=f"{display} — Mean Feature L2 Norm by Layer",
            save_path=os.path.join(out_dir, "feature_norms.png"),
        )

    # ── Combined accuracy-vs-depth plot (all models) ───────────────────────────
    visualize.plot_accuracy_vs_depth(
        depth_results_all,
        title="Layer-Wise Probe Accuracy vs Depth — All Models",
        save_path=os.path.join(config.RESULTS_DIR, "s45", "accuracy_vs_depth_all.png"),
    )

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = os.path.join(config.RESULTS_DIR, "s45", "depth_probing_table.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df = pd.DataFrame(all_summary_rows)
    df.to_csv(summary_path, index=False)
    print(f"\n[4.5] Summary saved to {summary_path}")
    print(df.to_string(index=False))

    return all_summary_rows


if __name__ == "__main__":
    run()
