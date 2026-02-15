from __future__ import annotations

import math
from typing import List

from .tensor import Tensor


class CrossEntropyLoss:
    def __init__(self) -> None:
        self.g16Probs: List[float] = []
        self.g16Labels: List[int] = []
        self.g16Batch = 0
        self.g16Classes = 0

    def G16Forward(self, logits: Tensor, labels: List[int]) -> float:
        if len(logits.shape) != 2:
            raise ValueError("CrossEntropyLoss expects logits with shape [batch, classes]")

        batch, classes = logits.shape
        if batch != len(labels):
            raise ValueError("Label count does not match batch size")

        self.g16Probs = [0.0 for _ in range(batch * classes)]
        self.g16Labels = list(labels)
        self.g16Batch = batch
        self.g16Classes = classes

        total_loss = 0.0
        for n_idx in range(batch):
            start = n_idx * classes
            row = logits.data[start : start + classes]
            max_logit = row[0]
            for value in row[1:]:
                if value > max_logit:
                    max_logit = value

            exps = [0.0 for _ in range(classes)]
            denom = 0.0
            for c_idx in range(classes):
                exp_val = math.exp(row[c_idx] - max_logit)
                exps[c_idx] = exp_val
                denom += exp_val

            for c_idx in range(classes):
                self.g16Probs[start + c_idx] = exps[c_idx] / denom

            target = labels[n_idx]
            prob_target = self.g16Probs[start + target]
            if prob_target < 1e-12:
                prob_target = 1e-12
            total_loss += -math.log(prob_target)

        return total_loss / batch

    def G16Backward(self) -> Tensor:
        grad = self.g16Probs[:]
        for n_idx, target in enumerate(self.g16Labels):
            grad[n_idx * self.g16Classes + target] -= 1.0

        scale = 1.0 / self.g16Batch
        for idx in range(len(grad)):
            grad[idx] *= scale

        return Tensor(grad, (self.g16Batch, self.g16Classes), requires_grad=False)
