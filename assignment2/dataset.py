"""
Dataset loading and splitting utilities for the AID (Aerial Images Dataset).

The dataset is in image-folder format:
  train_data/
    Airport/   *.jpg
    Beach/     *.jpg
    ...        (30 classes total)

Provides:
  - get_transforms()        : train/val augmentation pipelines
  - get_splits()            : stratified 80/20 train/val split
  - get_fewshot_subset()    : stratified subset for few-shot scenarios
  - get_fixed_pca_subset()  : fixed 30×30 sample indices for scenario 4.5 PCA
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

import config


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(train: bool, strong: bool = False) -> transforms.Compose:
    """Standard ImageNet-normalised transforms for train or val."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        if strong:
            return transforms.Compose([
                transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.80, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.01),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), ratio=(0.5, 2.0)),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class AIDDataset(Dataset):
    """
    AID image-folder dataset.
    Loads all (path, label) pairs up front; applies transform on __getitem__.
    """

    def __init__(self, samples: list, class_to_idx: dict, transform=None):
        self.samples = samples          # list of (img_path, label_idx)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def targets(self):
        return [s[1] for s in self.samples]

    @property
    def classes(self):
        return sorted(self.class_to_idx, key=lambda k: self.class_to_idx[k])


def _scan_folder(root: str):
    """Return (samples, class_to_idx) from an image-folder directory."""
    classes = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    samples = []
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))
    return samples, class_to_idx


def get_splits(seed: int = config.SEED):
    """
    Stratified 80/20 split of the AID dataset.
    Returns (train_dataset, val_dataset) as AIDDataset instances.
    """
    samples, class_to_idx = _scan_folder(config.DATA_ROOT)
    labels = [s[1] for s in samples]

    train_idx, val_idx = train_test_split(
        range(len(samples)),
        test_size=config.VAL_FRACTION,
        stratify=labels,
        random_state=seed,
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]

    train_ds = AIDDataset(train_samples, class_to_idx, transform=get_transforms(train=True))
    val_ds   = AIDDataset(val_samples,   class_to_idx, transform=get_transforms(train=False))

    print(f"Dataset split — Train: {len(train_ds)}, Val: {len(val_ds)} | Classes: {len(class_to_idx)}")
    return train_ds, val_ds


# ── Few-Shot Subset ───────────────────────────────────────────────────────────

def get_fewshot_subset(train_dataset: AIDDataset, fraction: float, seed: int = config.SEED):
    """
    Stratified fraction of the training set for few-shot experiments.
    Returns a torch.utils.data.Subset of train_dataset.
    """
    if fraction >= 1.0:
        return train_dataset

    labels = train_dataset.targets
    all_idx = list(range(len(train_dataset)))

    keep_idx, _ = train_test_split(
        all_idx,
        train_size=fraction,
        stratify=labels,
        random_state=seed,
    )
    subset = Subset(train_dataset, keep_idx)
    n = len(keep_idx)
    print(f"Few-shot subset ({fraction*100:.0f}%): {n} samples")
    return subset


# ── Fixed PCA Subset ──────────────────────────────────────────────────────────

def get_fixed_pca_subset(val_dataset: AIDDataset,
                         n_per_class: int = config.LAYER_PROBE_SUBSET_PER_CLASS,
                         seed: int = config.SEED):
    """
    Fixed subset of n_per_class samples per class from the validation set.
    Uses the same indices across ALL models/layers (called once, indices cached).
    Returns a Subset and the list of class labels for the subset.
    """
    rng = random.Random(seed)
    targets = val_dataset.targets
    class_to_indices = {}
    for i, t in enumerate(targets):
        class_to_indices.setdefault(t, []).append(i)

    chosen = []
    for cls_idx in sorted(class_to_indices):
        idx_list = class_to_indices[cls_idx]
        if len(idx_list) >= n_per_class:
            chosen.extend(rng.sample(idx_list, n_per_class))
        else:
            chosen.extend(idx_list)  # use all if fewer than n_per_class

    subset_labels = [targets[i] for i in chosen]
    subset = Subset(val_dataset, chosen)
    print(f"Fixed PCA subset: {len(chosen)} samples ({n_per_class} per class × {len(class_to_indices)} classes)")
    return subset, subset_labels


# ── DataLoader helpers ────────────────────────────────────────────────────────

def make_loader(dataset, batch_size: int = config.BATCH_SIZE,
                shuffle: bool = False, num_workers: int = config.NUM_WORKERS) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
