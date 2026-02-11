# layers/activation_relu.py

from layers.base_layer import BaseLayer
from core.tensor import Tensor


class ReLU(BaseLayer):
    """
    Channel-aware ReLU.
    """

    def forward(self, x):
        data = x.data

        # Scalar
        if isinstance(data, (int, float)):
            return Tensor(max(0.0, data), requires_grad=True)

        # 1D vector
        if isinstance(data[0], (int, float)):
            return Tensor([max(0.0, v) for v in data], requires_grad=True)

        # 3D tensor: [C][H][W]
        output = []
        for c in data:
            channel = []
            for row in c:
                channel.append([max(0.0, v) for v in row])
            output.append(channel)

        return Tensor(output, requires_grad=True)
