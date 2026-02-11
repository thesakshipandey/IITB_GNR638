# ops/elementwise.py

from core.tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Scalar addition.
    """
    out = Tensor(a.data + b.data, requires_grad=True)

    def backward_fn():
        if a.requires_grad:
            if a.grad is None:
                a.grad = out.grad
            else:
                a.grad += out.grad

        if b.requires_grad:
            if b.grad is None:
                b.grad = out.grad
            else:
                b.grad += out.grad

    out.set_backward(backward_fn, parents=[a, b])
    return out


def multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Scalar multiplication.
    """
    out = Tensor(a.data * b.data, requires_grad=True)

    def backward_fn():
        if a.requires_grad:
            grad_a = b.data * out.grad
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad += grad_a

        if b.requires_grad:
            grad_b = a.data * out.grad
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad += grad_b

    out.set_backward(backward_fn, parents=[a, b])
    return out


def relu(x: Tensor) -> Tensor:
    assert isinstance(x.data, (int, float)), "ReLU expects scalar Tensor"

    out_value = x.data if x.data > 0 else 0.0
    out = Tensor(out_value, requires_grad=True)

    def backward_fn():
        if x.requires_grad:
            grad_x = out.grad if x.data > 0 else 0.0
            x.grad = grad_x if x.grad is None else x.grad + grad_x

    out.set_backward(backward_fn, parents=[x])
    return out

