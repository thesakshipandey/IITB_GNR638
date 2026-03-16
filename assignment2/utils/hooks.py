"""
Forward-hook manager for feature extraction from intermediate layers.

Usage:
    extractor = FeatureExtractor(model, {"early": "layer1", "final": "layer4"})
    features = extractor.extract(loader, device)
    # features == {"early": np.ndarray(N, D), "final": np.ndarray(N, D)}
    extractor.remove_hooks()
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from functools import reduce
import operator


def _get_nested_module(model: nn.Module, path: str) -> nn.Module:
    """Retrieve a nested submodule by dot-separated path."""
    return reduce(getattr, path.split("."), model)


class FeatureExtractor:
    """
    Registers forward hooks on named layers and collects their output
    feature maps (spatially pooled to a vector) for an entire DataLoader.
    """

    def __init__(self, model: nn.Module, layer_map: dict):
        """
        Args:
            model:     nn.Module to hook into
            layer_map: {label: module_path} e.g. {"early": "layer1", "final": "layer4"}
        """
        self.model = model
        self.layer_map = layer_map      # label -> path
        self._hooks = []
        self._buffers = {label: [] for label in layer_map}

        for label, path in layer_map.items():
            module = _get_nested_module(model, path)
            hook = module.register_forward_hook(self._make_hook(label))
            self._hooks.append(hook)

    def _make_hook(self, label: str):
        def hook_fn(module, input, output):
            # output shape: (B, C, H, W) for conv layers, or (B, C) for linear
            x = output.detach()
            if x.dim() == 4:
                # Global average pooling over spatial dims
                x = x.mean(dim=[2, 3])    # (B, C)
            elif x.dim() == 3:
                x = x.mean(dim=1)          # (B, C) for some transformer-like outputs
            self._buffers[label].append(x.cpu())
        return hook_fn

    def extract(self, loader: DataLoader, device: torch.device) -> dict:
        """
        Run model in eval mode over loader; return dict of features and labels.

        Returns:
            {
              label: np.ndarray of shape (N, D),
              ...
              "__labels__": np.ndarray of shape (N,)
            }
        """
        # Clear buffers
        for label in self._buffers:
            self._buffers[label] = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                _ = self.model(images)
                all_labels.append(labels.numpy())

        result = {}
        for label in self.layer_map:
            result[label] = torch.cat(self._buffers[label], dim=0).numpy()
            self._buffers[label] = []  # free memory

        result["__labels__"] = np.concatenate(all_labels, axis=0)
        return result

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
