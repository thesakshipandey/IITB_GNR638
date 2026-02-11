# layers/base_layer.py

class BaseLayer:
    """
    Abstract base class for all neural network layers.
    """

    def forward(self, x):
        """
        Computes the forward pass of the layer.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Forward method not implemented.")

    def parameters(self):
        """
        Returns a list of trainable Tensor parameters.
        By default, layers have no parameters.
        """
        return []

    def zero_grad(self):
        """
        Resets gradients of all parameters in the layer.
        """
        for p in self.parameters():
            p.grad = None
