# core/autograd_engine.py

from core.graph import ComputationGraph


class AutogradEngine:
    """
    Executes reverse-mode automatic differentiation
    using an explicit computation graph.
    """

    def __init__(self):
        self.graph = ComputationGraph()

    def backward(self, output_tensor):
        """
        Performs backpropagation starting from the output tensor.
        """
        # Seed gradient at the output
        output_tensor.grad = 1.0

        # Build topological order
        topo_order = self.graph.build_topological_order(output_tensor)

        # Traverse in reverse topological order
        for tensor in reversed(topo_order):
            if tensor.backward_fn is not None:
                tensor.backward_fn()

    def register_tensor(self, tensor):
        """
        Registers a tensor with the computation graph.
        """
        self.graph.register(tensor)

    def clear(self):
        """
        Clears the computation graph.
        """
        self.graph.clear()
