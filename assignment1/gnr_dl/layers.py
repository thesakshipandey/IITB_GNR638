from __future__ import annotations

import random
from typing import List

from .backend import G16GetBackend
from .tensor import Parameter, Tensor


class Layer:
    def G16Forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def G16Backward(self, grad_output: Tensor) -> Tensor:
        raise NotImplementedError

    def G16Parameters(self) -> List[Parameter]:
        return []


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        limit = (1.0 / fan_in) ** 0.5
        weight_size = out_channels * in_channels * kernel_size * kernel_size
        weight_data = [random.uniform(-limit, limit) for _ in range(weight_size)]

        self.weight = Parameter(weight_data, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Parameter([0.0 for _ in range(out_channels)], (out_channels,))
        self.g16Input: Tensor | None = None

    def G16Forward(self, x: Tensor) -> Tensor:
        backend = G16GetBackend()
        batch, in_c, in_h, in_w = x.shape
        if in_c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {in_c}")

        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1

        out_data = backend.G16Conv2dForward(
            x.data,
            self.weight.data,
            self.bias.data,
            batch,
            in_c,
            in_h,
            in_w,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
        )
        self.g16Input = x
        return Tensor(out_data, (batch, self.out_channels, out_h, out_w), requires_grad=False)

    def G16Backward(self, grad_output: Tensor) -> Tensor:
        if self.g16Input is None:
            raise RuntimeError("Conv2D backward called before forward")

        backend = G16GetBackend()
        batch, in_c, in_h, in_w = self.g16Input.shape

        dx, dw, db = backend.G16Conv2dBackward(
            grad_output.data,
            self.g16Input.data,
            self.weight.data,
            batch,
            in_c,
            in_h,
            in_w,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
        )

        for idx in range(len(self.weight.grad)):
            self.weight.grad[idx] += dw[idx]
        for idx in range(len(self.bias.grad)):
            self.bias.grad[idx] += db[idx]

        return Tensor(dx, self.g16Input.shape, requires_grad=False)

    def G16Parameters(self) -> List[Parameter]:
        return [self.weight, self.bias]


class ReLU(Layer):
    def __init__(self) -> None:
        self.g16InputData: List[float] | None = None
        self.g16Shape = None

    def G16Forward(self, x: Tensor) -> Tensor:
        out_data = G16GetBackend().G16ReluForward(x.data)
        self.g16InputData = x.data
        self.g16Shape = x.shape
        return Tensor(out_data, x.shape, requires_grad=False)

    def G16Backward(self, grad_output: Tensor) -> Tensor:
        if self.g16InputData is None or self.g16Shape is None:
            raise RuntimeError("ReLU backward called before forward")
        dx = G16GetBackend().G16ReluBackward(grad_output.data, self.g16InputData)
        return Tensor(dx, self.g16Shape, requires_grad=False)


class MaxPool2D(Layer):
    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.g16Mask: List[int] | None = None
        self.g16InputShape = None

    def G16Forward(self, x: Tensor) -> Tensor:
        batch, channels, in_h, in_w = x.shape
        out_data, mask, out_h, out_w = G16GetBackend().G16Maxpool2dForward(
            x.data,
            batch,
            channels,
            in_h,
            in_w,
            self.pool_size,
            self.stride,
        )
        self.g16Mask = mask
        self.g16InputShape = x.shape
        return Tensor(out_data, (batch, channels, out_h, out_w), requires_grad=False)

    def G16Backward(self, grad_output: Tensor) -> Tensor:
        if self.g16Mask is None or self.g16InputShape is None:
            raise RuntimeError("MaxPool2D backward called before forward")
        input_size = 1
        for dim in self.g16InputShape:
            input_size *= dim

        dx = G16GetBackend().G16Maxpool2dBackward(grad_output.data, self.g16Mask, input_size)
        return Tensor(dx, self.g16InputShape, requires_grad=False)


class Flatten(Layer):
    def __init__(self) -> None:
        self.g16InputShape = None

    def G16Forward(self, x: Tensor) -> Tensor:
        self.g16InputShape = x.shape
        batch = x.shape[0]
        features = len(x.data) // batch
        return Tensor(x.data[:], (batch, features), requires_grad=False)

    def G16Backward(self, grad_output: Tensor) -> Tensor:
        if self.g16InputShape is None:
            raise RuntimeError("Flatten backward called before forward")
        return Tensor(grad_output.data[:], self.g16InputShape, requires_grad=False)


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

        limit = (1.0 / in_features) ** 0.5
        weight_data = [
            random.uniform(-limit, limit) for _ in range(out_features * in_features)
        ]
        self.weight = Parameter(weight_data, (out_features, in_features))
        self.bias = Parameter([0.0 for _ in range(out_features)], (out_features,))
        self.g16Input: Tensor | None = None

    def G16Forward(self, x: Tensor) -> Tensor:
        batch, features = x.shape
        if features != self.in_features:
            raise ValueError(
                f"Expected in_features={self.in_features}, received {features}"
            )

        out_data = G16GetBackend().G16LinearForward(
            x.data,
            self.weight.data,
            self.bias.data,
            batch,
            self.in_features,
            self.out_features,
        )
        self.g16Input = x
        return Tensor(out_data, (batch, self.out_features), requires_grad=False)

    def G16Backward(self, grad_output: Tensor) -> Tensor:
        if self.g16Input is None:
            raise RuntimeError("Linear backward called before forward")

        batch, _ = self.g16Input.shape
        dx, dw, db = G16GetBackend().G16LinearBackward(
            grad_output.data,
            self.g16Input.data,
            self.weight.data,
            batch,
            self.in_features,
            self.out_features,
        )

        for idx in range(len(self.weight.grad)):
            self.weight.grad[idx] += dw[idx]
        for idx in range(len(self.bias.grad)):
            self.bias.grad[idx] += db[idx]

        return Tensor(dx, self.g16Input.shape, requires_grad=False)

    def G16Parameters(self) -> List[Parameter]:
        return [self.weight, self.bias]
