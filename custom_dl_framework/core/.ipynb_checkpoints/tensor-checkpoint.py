# core/tensor.py

class Tensor:
    """
    Fundamental data structure of the framework.
    Stores data, gradient, and links to the computation graph.
    """

    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad

        # Gradient is initialized lazily
        self.grad = None

        # Graph-related attributes
        self.parents = []
        self.backward_fn = None

    def set_backward(self, backward_fn, parents):
        """
        Registers the backward function and parent tensors.
        """
        self.backward_fn = backward_fn
        self.parents = parents

    def zero_grad(self):
        """
        Clears gradient stored in this tensor.
        """
        self.grad = None

    def detach(self):
        """
        Returns a tensor disconnected from the computation graph.
        """
        return Tensor(self.data, requires_grad=False)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
