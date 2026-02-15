from __future__ import annotations

from typing import Dict, List, Tuple

from .layers import Conv2D, Flatten, Layer, Linear, MaxPool2D, ReLU
from .tensor import Parameter, Tensor


class SimpleCNN:
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        input_size: int = 32,
        conv_out_channels: int = 16,
        conv2_out_channels: int = 32,
        use_second_conv: bool = False,
        fc_hidden_features: int = 0,
        kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        pool_size: int = 2,
        pool_stride: int = 2,
    ) -> None:
        self.input_channels = input_channels
        self.input_size = input_size
        self.conv_out_channels = conv_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.use_second_conv = use_second_conv
        self.fc_hidden_features = fc_hidden_features
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.num_classes = num_classes

        conv1_out_h = (input_size + 2 * conv_padding - kernel_size) // conv_stride + 1
        conv1_out_w = (input_size + 2 * conv_padding - kernel_size) // conv_stride + 1
        pool1_out_h = (conv1_out_h - pool_size) // pool_stride + 1
        pool1_out_w = (conv1_out_w - pool_size) // pool_stride + 1

        if pool1_out_h <= 0 or pool1_out_w <= 0:
            raise ValueError("Invalid model configuration: pooled feature size is non-positive")

        if use_second_conv:
            conv2_out_h = (pool1_out_h + 2 * conv_padding - kernel_size) // conv_stride + 1
            conv2_out_w = (pool1_out_w + 2 * conv_padding - kernel_size) // conv_stride + 1
            pool2_out_h = (conv2_out_h - pool_size) // pool_stride + 1
            pool2_out_w = (conv2_out_w - pool_size) // pool_stride + 1
            if pool2_out_h <= 0 or pool2_out_w <= 0:
                raise ValueError(
                    "Invalid model configuration: second pooled feature size is non-positive"
                )
            self.final_feature_h = pool2_out_h
            self.final_feature_w = pool2_out_w
            self.final_feature_channels = conv2_out_channels
        else:
            self.final_feature_h = pool1_out_h
            self.final_feature_w = pool1_out_w
            self.final_feature_channels = conv_out_channels

        self.flatten_features = (
            self.final_feature_channels * self.final_feature_h * self.final_feature_w
        )

        self.layers: List[Layer] = [
            Conv2D(
                in_channels=input_channels,
                out_channels=conv_out_channels,
                kernel_size=kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            ),
            ReLU(),
            MaxPool2D(pool_size=pool_size, stride=pool_stride),
        ]

        if use_second_conv:
            self.layers.extend(
                [
                    Conv2D(
                        in_channels=conv_out_channels,
                        out_channels=conv2_out_channels,
                        kernel_size=kernel_size,
                        stride=conv_stride,
                        padding=conv_padding,
                    ),
                    ReLU(),
                    MaxPool2D(pool_size=pool_size, stride=pool_stride),
                ]
            )

        self.layers.append(Flatten())

        if fc_hidden_features > 0:
            self.layers.extend(
                [
                    Linear(in_features=self.flatten_features, out_features=fc_hidden_features),
                    ReLU(),
                    Linear(in_features=fc_hidden_features, out_features=num_classes),
                ]
            )
        else:
            self.layers.append(Linear(in_features=self.flatten_features, out_features=num_classes))

    def G16Forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer.G16Forward(out)
        return out

    def G16Backward(self, grad_output: Tensor) -> Tensor:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.G16Backward(grad)
        return grad

    def G16Parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for layer in self.layers:
            params.extend(layer.G16Parameters())
        return params

    def G16Complexity(self) -> Dict[str, int]:
        conv1_out_h = (self.input_size + 2 * self.conv_padding - self.kernel_size) // self.conv_stride + 1
        conv1_out_w = (self.input_size + 2 * self.conv_padding - self.kernel_size) // self.conv_stride + 1
        pool1_out_h = (conv1_out_h - self.pool_size) // self.pool_stride + 1
        pool1_out_w = (conv1_out_w - self.pool_size) // self.pool_stride + 1

        conv1_params = (
            self.conv_out_channels * self.input_channels * self.kernel_size * self.kernel_size
            + self.conv_out_channels
        )
        conv1_macs = (
            conv1_out_h
            * conv1_out_w
            * self.conv_out_channels
            * self.input_channels
            * self.kernel_size
            * self.kernel_size
        )

        total_params = conv1_params
        total_macs = conv1_macs
        extra_flops = 0

        conv1_output_elems = conv1_out_h * conv1_out_w * self.conv_out_channels
        extra_flops += conv1_output_elems
        extra_flops += conv1_output_elems

        pool1_output_elems = pool1_out_h * pool1_out_w * self.conv_out_channels
        extra_flops += pool1_output_elems * (self.pool_size * self.pool_size - 1)

        if self.use_second_conv:
            conv2_out_h = (pool1_out_h + 2 * self.conv_padding - self.kernel_size) // self.conv_stride + 1
            conv2_out_w = (pool1_out_w + 2 * self.conv_padding - self.kernel_size) // self.conv_stride + 1
            conv2_params = (
                self.conv2_out_channels
                * self.conv_out_channels
                * self.kernel_size
                * self.kernel_size
                + self.conv2_out_channels
            )
            conv2_macs = (
                conv2_out_h
                * conv2_out_w
                * self.conv2_out_channels
                * self.conv_out_channels
                * self.kernel_size
                * self.kernel_size
            )
            total_params += conv2_params
            total_macs += conv2_macs

            pool2_out_h = (conv2_out_h - self.pool_size) // self.pool_stride + 1
            pool2_out_w = (conv2_out_w - self.pool_size) // self.pool_stride + 1

            conv2_output_elems = conv2_out_h * conv2_out_w * self.conv2_out_channels
            extra_flops += conv2_output_elems
            extra_flops += conv2_output_elems

            pool2_output_elems = pool2_out_h * pool2_out_w * self.conv2_out_channels
            extra_flops += pool2_output_elems * (self.pool_size * self.pool_size - 1)
        else:
            conv2_out_h = 0
            conv2_out_w = 0

        if self.fc_hidden_features > 0:
            fc1_params = self.flatten_features * self.fc_hidden_features + self.fc_hidden_features
            fc2_params = self.fc_hidden_features * self.num_classes + self.num_classes
            fc1_macs = self.flatten_features * self.fc_hidden_features
            fc2_macs = self.fc_hidden_features * self.num_classes
            total_params += fc1_params + fc2_params
            total_macs += fc1_macs + fc2_macs

            extra_flops += self.fc_hidden_features
            extra_flops += self.fc_hidden_features
            extra_flops += self.num_classes
        else:
            fc_params = self.flatten_features * self.num_classes + self.num_classes
            fc_macs = self.flatten_features * self.num_classes
            total_params += fc_params
            total_macs += fc_macs
            extra_flops += self.num_classes

        total_flops = 2 * total_macs + extra_flops

        return {
            "parameters": total_params,
            "macs": total_macs,
            "flops": total_flops,
            "conv1_output_h": conv1_out_h,
            "conv1_output_w": conv1_out_w,
            "pool1_output_h": pool1_out_h,
            "pool1_output_w": pool1_out_w,
            "conv2_output_h": conv2_out_h,
            "conv2_output_w": conv2_out_w,
            "final_feature_h": self.final_feature_h,
            "final_feature_w": self.final_feature_w,
        }


def G16ResolveModelArch(model_arch: str, num_classes: int) -> str:
    requested = model_arch.strip().lower()
    if requested == "auto":
        if num_classes == 10:
            return "data1"
        if num_classes == 100:
            return "data2"
        return "custom"

    if requested not in {"custom", "data1", "data2"}:
        raise ValueError(
            f"Unsupported model_arch '{model_arch}'. Use one of: auto, custom, data1, data2"
        )
    return requested


def G16BuildModel(
    num_classes: int,
    input_channels: int,
    input_size: int,
    model_arch: str = "custom",
    conv_out_channels: int = 16,
    conv2_out_channels: int = 32,
    use_second_conv: bool = False,
    fc_hidden_features: int = 0,
    kernel_size: int = 3,
    conv_stride: int = 1,
    conv_padding: int = 1,
    pool_size: int = 2,
    pool_stride: int = 2,
) -> Tuple[SimpleCNN, str]:
    resolved_arch = G16ResolveModelArch(model_arch, num_classes)

    if resolved_arch == "data1":
        model = SimpleCNN(
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size,
            conv_out_channels=16,
            conv2_out_channels=32,
            use_second_conv=True,
            fc_hidden_features=128,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            pool_size=pool_size,
            pool_stride=pool_stride,
        )
        return model, resolved_arch

    if resolved_arch == "data2":
        model = SimpleCNN(
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size,
            conv_out_channels=24,
            conv2_out_channels=48,
            use_second_conv=True,
            fc_hidden_features=192,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            pool_size=pool_size,
            pool_stride=pool_stride,
        )
        return model, resolved_arch

    model = SimpleCNN(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
        conv_out_channels=conv_out_channels,
        conv2_out_channels=conv2_out_channels,
        use_second_conv=use_second_conv,
        fc_hidden_features=fc_hidden_features,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        conv_padding=conv_padding,
        pool_size=pool_size,
        pool_stride=pool_stride,
    )
    return model, resolved_arch


# Backward-compatible aliases
build_model = G16BuildModel
