# metrics/complexity_metrics.py

def count_parameters(model):
    """
    Counts total number of trainable parameters in the model.
    """
    count = 0
    for p in model.parameters():
        # scalar Tensor
        if isinstance(p.data, (int, float)):
            count += 1
        # list Tensor (e.g., kernel)
        elif isinstance(p.data, list):
            count += _count_list(p.data)
    return count


def _count_list(lst):
    """
    Recursively counts elements in nested lists.
    """
    total = 0
    for x in lst:
        if isinstance(x, list):
            total += _count_list(x)
        else:
            total += 1
    return total


def compute_conv2d_macs(input_h, input_w, kernel_size, stride=1):
    """
    Computes MACs for a single-channel Conv2D layer.
    """
    out_h = (input_h - kernel_size) // stride + 1
    out_w = (input_w - kernel_size) // stride + 1

    # For each output pixel:
    # kernel_size * kernel_size multiplications + additions
    macs_per_output = kernel_size * kernel_size

    total_macs = out_h * out_w * macs_per_output
    return total_macs


def compute_dense_macs(in_features, out_features):
    """
    Computes MACs for a Dense layer.
    """
    return in_features * out_features


def macs_to_flops(macs):
    """
    Converts MACs to FLOPs.
    1 MAC = 2 FLOPs (1 multiply + 1 add)
    """
    return 2 * macs
