"""
Central configuration for GNR638 Assignment 2.
All hyperparameters, paths, and experimental settings are defined here.
"""

import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 123

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(__file__), "train_data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 30
IMAGE_SIZE = 224          # All three backbones use 224×224
VAL_FRACTION = 0.20       # 80/20 stratified split

# ── Models ────────────────────────────────────────────────────────────────────
# timm model names for: ResNet50, EfficientNet-B0, ConvNeXt-Tiny
MODELS = ["resnet50", "efficientnet_b0", "convnext_tiny"]
MODEL_DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
}

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS_FULL = 30          # Scenarios 4.1, 4.2, 4.4 (backbone training)
EPOCHS_FEWSHOT = 20       # Scenario 4.3 few-shot settings
NUM_WORKERS = 4

# Learning rates
LR_PROBE = 0.001          # Linear probe / head-only training
LR_FINETUNE = 1e-4        # Fine-tuning strategies (4.2)
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0

# Optional early-stopping defaults (can be overridden per scenario)
EARLY_STOP_PATIENCE = 0   # 0 disables early stopping
EARLY_STOP_MIN_DELTA = 1e-4

# ── Scenario 4.2 Fine-Tuning ──────────────────────────────────────────────────
SELECTIVE_UNFREEZE_FRACTION = 0.20   # ≤20% of backbone parameters
S42_HEAD_LR = 1e-3                   # Shared across all 4.2 strategies
S42_BACKBONE_LR = 1e-4               # Shared across all 4.2 strategies
S42_SELECTIVE_CALIBRATION_BATCHES = 3

# ── Scenario 4.3 Few-Shot ─────────────────────────────────────────────────────
FEWSHOT_FRACTIONS = [1.0, 0.50, 0.20, 0.05]
S43_STRONG_AUG = False
S43_WEIGHT_DECAY = 0.0001
S43_EARLY_STOP_PATIENCE = 0
S43_LOW_DATA_LR = 0.001

# ── Scenario 4.4 Corruptions ─────────────────────────────────────────────────
CORRUPTION_SIGMAS = [0.05, 0.1, 0.2]     # Gaussian noise σ values
MOTION_BLUR_KERNEL = 15                   # kernel size for motion blur
BRIGHTNESS_FACTOR = 1.5                   # factor for brightness shift

# ── Scenario 4.5 Layer Probing ────────────────────────────────────────────────
LAYER_PROBE_SUBSET_PER_CLASS = 30         # 30 classes × 30 samples = 900 for PCA

# Layer hooks per model (module attribute path strings for timm models)
LAYER_HOOKS = {
    "resnet50": {
        "early":  "layer1",
        "middle": "layer2",
        "final":  "layer4",
    },
    "efficientnet_b0": {
        "early":  "blocks.0",
        "middle": "blocks.3",
        "final":  "blocks.6",
    },
    "convnext_tiny": {
        "early":  "stages.0",
        "middle": "stages.1",
        "final":  "stages.3",
    },
}
