from __future__ import annotations

from typing import List

from .tensor import Tensor


def G16BatchAccuracy(logits: Tensor, labels: List[int]) -> float:
    batch, classes = logits.shape
    if batch == 0:
        return 0.0

    correct = 0
    for n_idx in range(batch):
        row_start = n_idx * classes
        best_class = 0
        best_score = logits.data[row_start]
        for c_idx in range(1, classes):
            score = logits.data[row_start + c_idx]
            if score > best_score:
                best_score = score
                best_class = c_idx
        if best_class == labels[n_idx]:
            correct += 1

    return correct / batch


batch_accuracy = G16BatchAccuracy
