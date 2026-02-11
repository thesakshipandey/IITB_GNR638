# core/grad_utils.py

def zero_grad(tensors):
    """
    Resets gradients of a list of tensors.
    """
    for t in tensors:
        t.grad = None


def clip_gradients(tensors, max_norm):
    """
    Clips gradients to prevent exploding gradients.
    """
    for t in tensors:
        if t.grad is None:
            continue
        if t.grad > max_norm:
            t.grad = max_norm
        elif t.grad < -max_norm:
            t.grad = -max_norm


def detach_tensor(tensor):
    """
    Returns a detached version of the tensor.
    """
    from core.tensor import Tensor
    return Tensor(tensor.data, requires_grad=False)
