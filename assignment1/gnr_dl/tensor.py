from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def G16Numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


@dataclass
class Tensor:
    data: List[float]
    shape: Tuple[int, ...]
    requires_grad: bool = False

    def __post_init__(self) -> None:
        if len(self.data) != G16Numel(self.shape):
            raise ValueError(
                f"Tensor data length {len(self.data)} does not match shape {self.shape}"
            )
        if self.requires_grad:
            self.grad = [0.0 for _ in range(len(self.data))]
        else:
            self.grad = []

    @classmethod
    def G16Zeros(cls, shape: Sequence[int], requires_grad: bool = False) -> "Tensor":
        return cls([0.0 for _ in range(G16Numel(shape))], tuple(shape), requires_grad)

    @classmethod
    def G16FromIterable(
        cls,
        values: Iterable[float],
        shape: Sequence[int],
        requires_grad: bool = False,
    ) -> "Tensor":
        return cls([float(v) for v in values], tuple(shape), requires_grad)

    @property
    def G16NumelProp(self) -> int:
        return len(self.data)

    def G16ZeroGrad(self) -> None:
        if self.requires_grad:
            for idx in range(len(self.grad)):
                self.grad[idx] = 0.0


class Parameter(Tensor):
    def __init__(self, data: List[float], shape: Sequence[int]):
        super().__init__(data, tuple(shape), requires_grad=True)
