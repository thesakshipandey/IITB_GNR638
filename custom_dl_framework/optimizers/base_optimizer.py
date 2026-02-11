# optimizers/base_optimizer.py

class BaseOptimizer:
    """
    Abstract base class for all optimizers.
    """

    def step(self):
        """
        Updates parameters based on their gradients.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("step() not implemented.")

    def zero_grad(self):
        """
        Clears gradients of all parameters.
        """
        raise NotImplementedError("zero_grad() not implemented.")
