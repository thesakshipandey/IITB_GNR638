# layers/conv2d.py

import random
from core.tensor import Tensor
from layers.base_layer import BaseLayer


class Conv2D(BaseLayer):
    """
    Minimal multi-channel Conv2D.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # kernels[o][c][i][j]
        self.kernels = [
            [
                [
                    [
                        Tensor(random.uniform(-0.1, 0.1), requires_grad=True)
                        for _ in range(kernel_size)
                    ]
                    for _ in range(kernel_size)
                ]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]

    def forward(self, x):
        # x.data shape: [C][H][W]
        C, H, W = len(x.data), len(x.data[0]), len(x.data[0][0])
        k = self.kernel_size
        s = self.stride

        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        output = [
            [[0.0 for _ in range(out_w)] for _ in range(out_h)]
            for _ in range(self.out_channels)
        ]

        for o in range(self.out_channels):
            for c in range(self.in_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        for ki in range(k):
                            for kj in range(k):
                                output[o][i][j] += (
                                    x.data[c][i * s + ki][j * s + kj]
                                    * self.kernels[o][c][ki][kj].data
                                )

        return Tensor(output, requires_grad=True)

    def parameters(self):
        params = []
        for o in range(self.out_channels):
            for c in range(self.in_channels):
                for row in self.kernels[o][c]:
                    params.extend(row)
        return params
