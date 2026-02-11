# data/dataset.py

import os
from core.tensor import Tensor
from runtime.profiler import Profiler
from data.image_reader import read_image


class ImageFolderDataset:
    """
    Dataset loader for folder-per-class image datasets.

    Expected structure:
        root/
            0/
                img1.png
                img2.png
            1/
                img3.png
            2/
                ...
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.inputs = []
        self.targets = []
        self.load_time = 0.0

        self._load_dataset()

    def _load_dataset(self):
        profiler = Profiler()
        profiler.start()

        # Ensure deterministic order
        class_dirs = sorted(os.listdir(self.root_dir))

        for class_name in class_dirs:
            class_path = os.path.join(self.root_dir, class_name)

            # Skip non-directories
            if not os.path.isdir(class_path):
                continue

            # Folder name is the label
            try:
                label = int(class_name)
            except ValueError:
                # Skip folders that are not valid class labels
                continue

            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)

                # Basic file check
                if not os.path.isfile(file_path):
                    continue

                # Read image (as 2D list)
                image_data = read_image(file_path)

                # Wrap image in Tensor
                image_tensor = Tensor(image_data, requires_grad=False)

                self.inputs.append(image_tensor)
                self.targets.append(label)

        profiler.stop()
        self.load_time = profiler.get_elapsed()

    def __len__(self):
        return len(self.inputs)

    def get_data(self):
        """
        Returns all inputs and targets.
        """
        return self.inputs, self.targets

    def get_loading_time(self):
        """
        Returns dataset loading time in seconds.
        """
        return self.load_time
