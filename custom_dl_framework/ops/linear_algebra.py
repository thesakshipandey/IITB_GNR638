# ops/linear_algebra.py

from core.tensor import Tensor


def dot(a, b):
    """
    Dot product between two 1D tensors (lists).
    """
    assert len(a.data) == len(b.data), "Dot product dimension mismatch"

    result = 0.0
    for i in range(len(a.data)):
        result += a.data[i] * b.data[i]

    out = Tensor(result, requires_grad=True)

    def backward_fn():
        if a.requires_grad:
            grad_a = [b.data[i] * out.grad for i in range(len(b.data))]
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad = [a.grad[i] + grad_a[i] for i in range(len(grad_a))]

        if b.requires_grad:
            grad_b = [a.data[i] * out.grad for i in range(len(a.data))]
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad = [b.grad[i] + grad_b[i] for i in range(len(grad_b))]

    out.set_backward(backward_fn, parents=[a, b])
    return out


def matvec(matrix, vector):
    """
    Matrix-vector multiplication.
    matrix: list of lists
    vector: Tensor containing 1D list
    """
    output = []

    for row in matrix:
        s = 0.0
        for i in range(len(row)):
            s += row[i] * vector.data[i]
        output.append(s)

    out = Tensor(output, requires_grad=True)

    def backward_fn():
        if vector.requires_grad:
            grad_vec = [0.0 for _ in range(len(vector.data))]
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    grad_vec[j] += matrix[i][j] * out.grad[i]

            if vector.grad is None:
                vector.grad = grad_vec
            else:
                vector.grad = [
                    vector.grad[i] + grad_vec[i] for i in range(len(grad_vec))
                ]

    out.set_backward(backward_fn, parents=[vector])
    return out
