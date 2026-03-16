"""
Evaluation utilities for GNR638 Assignment 2.

Provides:
  - compute_confusion_matrix()  : full confusion matrix as numpy array
  - compute_all_metrics()       : top-1 accuracy and per-class accuracy dict
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import config


def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """
    Compute a num_classes × num_classes confusion matrix.
    Returns a raw-count matrix (not normalised).
    """
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[true, pred] += 1

    return cm


def compute_all_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    """
    Compute top-1 overall accuracy and per-class accuracy.

    Returns:
        {
          'top1_acc': float,
          'per_class_acc': {class_idx: float, ...},
        }
    """
    cm = compute_confusion_matrix(model, loader, num_classes, device)
    overall_correct = np.trace(cm)
    overall_total   = cm.sum()
    top1_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    per_class_acc = {}
    for c in range(num_classes):
        row_total = cm[c].sum()
        per_class_acc[c] = cm[c, c] / row_total if row_total > 0 else 0.0

    return {"top1_acc": top1_acc, "per_class_acc": per_class_acc}
