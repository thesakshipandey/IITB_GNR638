# layers/dense.py

import random
from core.tensor import Tensor
from layers.base_layer import BaseLayer
from ops.linear_algebra import dot


class Dense(BaseLayer):
    """
    Fully connected (linear) layer.
    """

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix: out_features x in_features
        self.weights = [
            [Tensor(random.uniform(-0.1, 0.1), requires_grad=True)
             for _ in range(in_features)]
            for _ in range(out_features)
        ]

        # Bias vector
        self.biases = [
            Tensor(0.0, requires_grad=True) for _ in range(out_features)
        ]

    def forward(self, x):
        """
        Forward pass.
        x: Tensor containing a 1D list of length in_features
        """
        output = []

        for i in range(self.out_features):
            # Convert weight row into Tensor for dot product
            w_row = Tensor([w.data for w in self.weights[i]],
                           requires_grad=False)

            s = dot(w_row, x)
            s = Tensor(s.data + self.biases[i].data, requires_grad=True)

            # Manually connect graph for bias addition
            def backward_fn(s=s, w_row=w_row, x=x, b=self.biases[i]):
                if b.requires_grad:
                    b.grad = s.grad if b.grad is None else b.grad + s.grad

            s.set_backward(backward_fn, parents=[x, self.biases[i]])
            output.append(s)

        return output

    def parameters(self):
        """
        Returns all trainable parameters.
        """
        params = []
        for row in self.weights:
            params.extend(row)
        params.extend(self.biases)
        return params
