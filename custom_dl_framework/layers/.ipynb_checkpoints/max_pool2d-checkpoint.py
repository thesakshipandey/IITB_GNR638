# layers/max_pool2d.py

from core.tensor import Tensor
from layers.base_layer import BaseLayer


class MaxPool2D(BaseLayer):
    """
    Channel-wise MaxPool2D.
    """

    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        # x.data shape: [C][H][W]
        C = len(x.data)
        H = len(x.data[0])
        W = len(x.data[0][0])

        P = self.pool_size
        S = self.stride

        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        output = []

        for c in range(C):
            channel = []
            for i in range(H_out):
                row = []
                for j in range(W_out):
                    max_val = float("-inf")
                    for pi in range(P):
                        for pj in range(P):
                            val = x.data[c][i * S + pi][j * S + pj]
                            if val > max_val:
                                max_val = val
                    row.append(max_val)
                channel.append(row)
            output.append(channel)

        return Tensor(output, requires_grad=True)
