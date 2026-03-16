"""
Evaluation-time corruption transforms for Scenario 4.4.

All transforms are torchvision-compatible (operate on PIL Images or Tensors)
and are applied ONLY at evaluation time, never during training.

Corruptions:
  - GaussianNoise(sigma)          : pixel-level Gaussian noise (tensor-level)
  - MotionBlur(kernel_size)       : directional motion blur (PIL-level)
  - BrightnessShift(factor)       : brightness adjustment (PIL-level)
"""

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter


class GaussianNoise:
    """
    Add Gaussian noise to a normalised tensor.
    Applied after ToTensor() + Normalize() so noise is in normalised space.

    Args:
        sigma: standard deviation of the noise
    """
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise

    def __repr__(self):
        return f"GaussianNoise(sigma={self.sigma})"


class MotionBlur:
    """
    Apply horizontal motion blur using a box filter kernel (PIL-based).
    Applied before ToTensor().

    Args:
        kernel_size: length of the motion blur kernel (odd number)
    """
    def __init__(self, kernel_size: int = 15):
        self.kernel_size = kernel_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # Horizontal motion blur via numpy averaging over a horizontal strip
        arr = np.array(img, dtype=np.float32)
        k = self.kernel_size
        pad = k // 2
        result = np.zeros_like(arr)
        padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode='edge')
        for i in range(k):
            result += padded[:, i:i + arr.shape[1], :]
        result /= k
        return Image.fromarray(result.clip(0, 255).astype(np.uint8))

    def __repr__(self):
        return f"MotionBlur(kernel_size={self.kernel_size})"


class BrightnessShift:
    """
    Adjust image brightness (PIL-based, applied before ToTensor).

    Args:
        factor: brightness factor. 0=black, 1=original, >1=brighter.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        return TF.adjust_brightness(img, self.factor)

    def __repr__(self):
        return f"BrightnessShift(factor={self.factor})"


# ── Corruption transform factories ────────────────────────────────────────────

def get_corrupted_transforms(corruption_name: str, level=None,
                             base_size: int = 224) -> "torchvision.transforms.Compose":
    """
    Return a full eval transform pipeline with the specified corruption injected.

    For PIL-level corruptions (MotionBlur, BrightnessShift): inject before ToTensor.
    For tensor-level corruptions (GaussianNoise): inject after Normalize.

    Args:
        corruption_name: 'gaussian_noise', 'motion_blur', 'brightness_shift'
        level:           corruption parameter (sigma for gaussian; factor for brightness)
        base_size:       image resize/crop size
    """
    from torchvision import transforms

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if corruption_name == "gaussian_noise":
        sigma = level if level is not None else 0.1
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(base_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            GaussianNoise(sigma),
        ])

    elif corruption_name == "motion_blur":
        kernel_size = int(level) if level is not None else 15
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(base_size),
            MotionBlur(kernel_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif corruption_name == "brightness_shift":
        factor = level if level is not None else 1.5
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(base_size),
            BrightnessShift(factor),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
