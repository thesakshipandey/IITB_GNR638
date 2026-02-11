# metrics/accuracy.py

def accuracy(logits, target):
    """
    Computes accuracy for a single sample.

    logits: list of Tensor (raw model outputs)
    target: int (ground-truth class index)
    """
    # Argmax over logits
    max_index = 0
    max_value = logits[0].data

    for i in range(1, len(logits)):
        if logits[i].data > max_value:
            max_value = logits[i].data
            max_index = i

    return 1 if max_index == target else 0