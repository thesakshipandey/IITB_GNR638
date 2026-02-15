from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError as exc:  
    cv2 = None
    g16Cv2ImportError = exc

from .tensor import Tensor


def G16IsImageFile(filename: str) -> bool:
    return filename.lower().endswith(".png")


def G16ImageToChwFloatList(image, image_size: int) -> List[float]:
    data: List[float] = []
    for channel in range(3):
        for y_idx in range(image_size):
            for x_idx in range(image_size):
                data.append(float(image[y_idx][x_idx][channel]) / 255.0)
    return data


class ImageFolderDataset:
    def __init__(
        self,
        root_dir: str,
        image_size: int = 32,
        class_to_idx: Optional[Dict[str, int]] = None,
        preload: bool = True,
    ) -> None:
        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required for dataset loading in this assignment"
            ) from g16Cv2ImportError

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        self.root_dir = root_dir
        self.image_size = image_size
        self.preload = preload
        self.samples: List[List[float]] = []
        self.labels: List[int] = []
        self.paths: List[str] = []

        start_time = time.perf_counter()

        if class_to_idx is None:
            class_names = []
            for entry in sorted(os.listdir(root_dir)):
                full_path = os.path.join(root_dir, entry)
                if os.path.isdir(full_path):
                    class_names.append(entry)
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        else:
            self.class_to_idx = dict(class_to_idx)

        if not self.class_to_idx:
            raise ValueError(f"No class folders found under {root_dir}")

        self.idx_to_class = [None for _ in range(len(self.class_to_idx))]
        for class_name, class_idx in self.class_to_idx.items():
            self.idx_to_class[class_idx] = class_name

        ordered_classes: Sequence[Tuple[str, int]] = sorted(
            self.class_to_idx.items(), key=lambda item: item[1]
        )

        for class_name, class_idx in ordered_classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in sorted(os.listdir(class_dir)):
                if not G16IsImageFile(filename):
                    continue
                image_path = os.path.join(class_dir, filename)

                if preload:
                    tensor_data = self.G16ReadImage(image_path)
                    self.samples.append(tensor_data)
                else:
                    self.paths.append(image_path)

                self.labels.append(class_idx)

        self.loading_time_seconds = time.perf_counter() - start_time
        self.num_classes = len(self.class_to_idx)

        if len(self.labels) == 0:
            raise ValueError(f"No images found under {root_dir}")

    def G16ReadImage(self, image_path: str) -> List[float]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return G16ImageToChwFloatList(image, self.image_size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[List[float], int]:
        if self.preload:
            return self.samples[index], self.labels[index]

        image_data = self.G16ReadImage(self.paths[index])
        return image_data, self.labels[index]


class SubsetDataset:
    def __init__(self, dataset: ImageFolderDataset, indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.image_size = dataset.image_size
        self.num_classes = dataset.num_classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[List[float], int]:
        return self.dataset[self.indices[index]]


def G16TrainValSplit(
    dataset: ImageFolderDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[SubsetDataset, SubsetDataset]:
    train_set, val_set, _ = G16TrainValTestSplit(
        dataset=dataset,
        val_fraction=val_fraction,
        test_fraction=0.0,
        seed=seed,
    )
    return train_set, val_set


def G16TrainValTestSplit(
    dataset: ImageFolderDataset,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[SubsetDataset, SubsetDataset, SubsetDataset]:
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be non-negative")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(n * (1.0 - val_fraction - test_fraction))
    val_end = train_end + int(n * val_fraction)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (
        SubsetDataset(dataset, train_indices),
        SubsetDataset(dataset, val_indices),
        SubsetDataset(dataset, test_indices),
    )


class DataLoader:
    def __init__(self, dataset: ImageFolderDataset, batch_size: int, shuffle: bool = True) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.g16Indices: List[int] = []
        self.g16Cursor = 0

    def __iter__(self) -> "DataLoader":
        self.g16Indices = [idx for idx in range(len(self.dataset))]
        if self.shuffle:
            random.shuffle(self.g16Indices)
        self.g16Cursor = 0
        return self

    def __next__(self) -> Tuple[Tensor, List[int]]:
        if self.g16Cursor >= len(self.g16Indices):
            raise StopIteration

        batch_indices = self.g16Indices[self.g16Cursor : self.g16Cursor + self.batch_size]
        self.g16Cursor += self.batch_size

        batch_data: List[float] = []
        batch_labels: List[int] = []

        for idx in batch_indices:
            sample, label = self.dataset[idx]
            batch_data.extend(sample)
            batch_labels.append(label)

        batch = len(batch_indices)
        channels = 3
        image_size = self.dataset.image_size

        x_batch = Tensor(
            batch_data,
            (batch, channels, image_size, image_size),
            requires_grad=False,
        )
        return x_batch, batch_labels
