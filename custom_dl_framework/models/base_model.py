# models/base_model.py

class BaseModel:
    """
    Base class for all models.
    """

    def forward(self, x):
        """
        Forward pass through the model.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("forward() not implemented.")

    def parameters(self):
        """
        Returns all trainable parameters of the model.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        """
        Resets gradients of all model parameters.
        """
        for p in self.parameters():
            p.grad = None
