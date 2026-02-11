# data/batch_sampler.py

class BatchSampler:
    """
    Simple batch sampler for datasets.
    """

    def __init__(self, inputs, targets, batch_size=1, shuffle=True):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(inputs)

    def __iter__(self):
        indices = list(range(self.num_samples))

        if self.shuffle:
            self._shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_inputs = [self.inputs[i] for i in batch_indices]
            batch_targets = [self.targets[i] for i in batch_indices]
            yield batch_inputs, batch_targets

    def _shuffle(self, indices):
        """
        In-place Fisher–Yates shuffle (no random library required).
        """
        import random
        for i in range(len(indices) - 1, 0, -1):
            j = random.randint(0, i)
            indices[i], indices[j] = indices[j], indices[i]
