"""
Shared training and validation loops for GNR638 Assignment 2.

Provides:
  - train_epoch()   : single training epoch with optional gradient norm logging
  - validate()      : evaluation pass (no gradients)
  - run_training()  : full training loop returning history dict
  - set_seed()      : global seed setter for reproducibility
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

import config


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED) -> None:
    """Set all relevant random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Single epoch ──────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_norm_log: bool = False,
) -> tuple:
    """
    Run one training epoch.

    Returns:
        (avg_loss, accuracy, grad_norms_dict)
        grad_norms_dict: {param_group_prefix: mean_norm} if grad_norm_log else {}
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_norms_accumulator = {}  # prefix -> list of norms

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # ── Gradient norm logging ─────────────────────────────────────────────
        if grad_norm_log:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Group by top-level module name
                    prefix = name.split(".")[0]
                    norm = param.grad.norm().item()
                    grad_norms_accumulator.setdefault(prefix, []).append(norm)

        if config.GRAD_CLIP_NORM is not None and config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.GRAD_CLIP_NORM,
            )

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    # Average norms per group
    grad_norms = {
        k: float(np.mean(v)) for k, v in grad_norms_accumulator.items()
    } if grad_norm_log else {}

    return avg_loss, accuracy, grad_norms


# ── Validation ────────────────────────────────────────────────────────────────

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate model on loader without gradient computation.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── Full training loop ────────────────────────────────────────────────────────

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    grad_norm_log: bool = False,
    verbose: bool = True,
    early_stop_patience: int = config.EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = config.EARLY_STOP_MIN_DELTA,
) -> dict:
    """
    Run the full training loop.

    Returns a history dict:
      {
        'train_loss': [...],
        'train_acc':  [...],
        'val_loss':   [...],
        'val_acc':    [...],
        'grad_norms': [{prefix: norm}, ...],  # one dict per epoch if grad_norm_log
        'epoch_times': [...],                 # wall-clock seconds per epoch
        'best_val_acc': float,
        'best_epoch': int,
        'best_state_dict': {param_name: tensor},
      }
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc": [],
        "grad_norms": [], "epoch_times": [],
    }
    best_val_acc = float("-inf")
    best_epoch = -1
    best_state_dict = None
    best_train_acc = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, grad_norms = train_epoch(
            model, train_loader, optimizer, criterion, device, grad_norm_log
        )
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["grad_norms"].append(grad_norms)
        history["epoch_times"].append(elapsed)

        if va_acc > (best_val_acc + early_stop_min_delta):
            best_val_acc = va_acc
            best_epoch = epoch
            best_train_acc = tr_acc
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if verbose:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

        if early_stop_patience and epoch >= 3 and epochs_without_improve >= early_stop_patience:
            if verbose:
                print(
                    f"  Early stopping triggered at epoch {epoch} "
                    f"(no val-acc improvement in {early_stop_patience} epochs)."
                )
            break

    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["best_state_dict"] = best_state_dict
    history["best_train_acc"] = best_train_acc
    history["stopped_early"] = len(history["val_acc"]) < epochs
    return history


# ── Optimizer factory ─────────────────────────────────────────────────────────

def make_optimizer_scheduler(
    model: nn.Module,
    lr: float,
    epochs: int,
    weight_decay: float = config.WEIGHT_DECAY,
):
    """
    Create AdamW optimizer (only over trainable params) + CosineAnnealingLR.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler
