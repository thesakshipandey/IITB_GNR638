# ops/convolution.py

from core.tensor import Tensor


def conv2d(input_tensor, kernel, stride=1):
    """
    2D convolution (single-channel, no padding).

    input_tensor.data : 2D list (H x W)
    kernel.data       : 2D list (kH x kW)
    """

    input_data = input_tensor.data
    kernel_data = kernel.data

    in_h = len(input_data)
    in_w = len(input_data[0])

    k_h = len(kernel_data)
    k_w = len(kernel_data[0])

    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1

    output = [[0.0 for _ in range(out_w)] for _ in range(out_h)]

    # -------- forward pass --------
    for i in range(out_h):
        for j in range(out_w):
            s = 0.0
            for ki in range(k_h):
                for kj in range(k_w):
                    s += (
                        input_data[i * stride + ki][j * stride + kj]
                        * kernel_data[ki][kj]
                    )
            output[i][j] = s

    out = Tensor(output, requires_grad=True)

    # -------- backward pass --------
    def backward_fn():
        # Gradient w.r.t input
        if input_tensor.requires_grad:
            if input_tensor.grad is None:
                input_tensor.grad = [
                    [0.0 for _ in range(in_w)] for _ in range(in_h)
                ]

            for i in range(out_h):
                for j in range(out_w):
                    for ki in range(k_h):
                        for kj in range(k_w):
                            input_tensor.grad[i * stride + ki][j * stride + kj] += (
                                kernel_data[ki][kj] * out.grad[i][j]
                            )

        # Gradient w.r.t kernel
        if kernel.requires_grad:
            if kernel.grad is None:
                kernel.grad = [
                    [0.0 for _ in range(k_w)] for _ in range(k_h)
                ]

            for ki in range(k_h):
                for kj in range(k_w):
                    for i in range(out_h):
                        for j in range(out_w):
                            kernel.grad[ki][kj] += (
                                input_data[i * stride + ki][j * stride + kj]
                                * out.grad[i][j]
                            )

    out.set_backward(backward_fn, parents=[input_tensor, kernel])
    return out
