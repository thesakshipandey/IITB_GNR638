from __future__ import annotations

from typing import Dict, List

from .tensor import Parameter


class SGD:
    def __init__(
        self,
        parameters: List[Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.g16Velocity: Dict[int, List[float]] = {}

        if momentum > 0.0:
            for param in parameters:
                self.g16Velocity[id(param)] = [0.0 for _ in range(len(param.data))]

    def G16ZeroGrad(self) -> None:
        for param in self.parameters:
            param.G16ZeroGrad()

    def G16Step(self) -> None:
        if self.momentum > 0.0:
            self.G16StepWithMomentum()
        else:
            self.G16StepWithoutMomentum()

    def G16StepWithoutMomentum(self) -> None:
        for param in self.parameters:
            for idx in range(len(param.data)):
                grad = param.grad[idx]
                if self.weight_decay > 0.0:
                    grad += self.weight_decay * param.data[idx]
                param.data[idx] -= self.lr * grad

    def G16StepWithMomentum(self) -> None:
        for param in self.parameters:
            velocity = self.g16Velocity[id(param)]
            for idx in range(len(param.data)):
                grad = param.grad[idx]
                if self.weight_decay > 0.0:
                    grad += self.weight_decay * param.data[idx]
                velocity[idx] = self.momentum * velocity[idx] + grad
                param.data[idx] -= self.lr * velocity[idx]
