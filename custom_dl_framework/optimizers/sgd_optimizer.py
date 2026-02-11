# optimizers/sgd_optimizer.py

from optimizers.base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        """
        Updates parameters using gradient descent.
        """
        for p in self.parameters:
            if p.grad is None:
                continue
            p.data -= self.learning_rate * p.grad

    def zero_grad(self):
        """
        Clears gradients of all parameters.
        """
        for p in self.parameters:
            p.grad = None
