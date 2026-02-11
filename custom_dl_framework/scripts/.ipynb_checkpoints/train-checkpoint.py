# scripts/train.py

import argparse

import config.default_config as cfg

from data.dataset import ImageFolderDataset
from data.batch_sampler import BatchSampler

from models.cnn_classifier import CNNClassifier
from losses.cross_entropy_loss import CrossEntropyLoss
from optimizers.sgd_optimizer import SGD
from runtime.trainer import Trainer

from metrics.complexity_metrics import (
    count_parameters,
    compute_conv2d_macs,
    compute_dense_macs,
    macs_to_flops
)


def train_val_split(inputs, targets, val_ratio):
    """
    Splits dataset into train and validation sets.
    Used ONLY for local experimentation.
    """
    split_idx = int(len(inputs) * (1 - val_ratio))
    return (
        inputs[:split_idx],
        targets[:split_idx],
        inputs[split_idx:],
        targets[split_idx:]
    )


def main(args):
    # ---------------- Load dataset ----------------
    dataset = ImageFolderDataset(args.data_path)
    inputs, targets = dataset.get_data()

    print(f"Dataset loading time: {dataset.get_loading_time():.2f} seconds")
    print(f"Total samples: {len(inputs)}")

    # ---------------- Optional validation split ----------------
    if cfg.USE_VALIDATION:
        train_x, train_y, val_x, val_y = train_val_split(
            inputs, targets, cfg.VALIDATION_SPLIT
        )
        print(
            f"Using validation split: "
            f"{len(train_x)} train / {len(val_x)} val"
        )
    else:
        train_x, train_y = inputs, targets
        val_x, val_y = None, None
        print("Training on full dataset (no validation split)")

    # ---------------- Model ----------------
    model = CNNClassifier(num_classes=cfg.NUM_CLASSES)

    # ---------------- Loss & Optimizer ----------------
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(
        parameters=model.parameters(),
        learning_rate=cfg.LEARNING_RATE
    )

    trainer = Trainer(model, loss_fn, optimizer)

    # ---------------- Complexity metrics ----------------
    param_count = count_parameters(model)

    conv_macs = compute_conv2d_macs(
        input_h=cfg.IMAGE_SIZE[0],
        input_w=cfg.IMAGE_SIZE[1],
        kernel_size=cfg.KERNEL_SIZE,
        stride=cfg.STRIDE
    )

    dense_macs = compute_dense_macs(
        in_features=225,
        out_features=cfg.NUM_CLASSES
    )

    total_macs = conv_macs + dense_macs
    total_flops = macs_to_flops(total_macs)

    print("Model complexity:")
    print(f"  Parameters: {param_count}")
    print(f"  MACs / forward: {total_macs}")
    print(f"  FLOPs / forward: {total_flops}")

    # ---------------- Training loop ----------------
    for epoch in range(cfg.EPOCHS):
        sampler = BatchSampler(
            train_x,
            train_y,
            batch_size=cfg.BATCH_SIZE,
            shuffle=cfg.SHUFFLE
        )

        epoch_inputs = []
        epoch_targets = []

        # Trainer works sample-wise → flatten batches
        for xb, yb in sampler:
            epoch_inputs.extend(xb)
            epoch_targets.extend(yb)

        stats = trainer.train_epoch(epoch_inputs, epoch_targets)

        print(
            f"Epoch [{epoch + 1}/{cfg.EPOCHS}] | "
            f"Loss: {stats['loss']:.4f} | "
            f"Accuracy: {stats['accuracy'] * 100:.2f}% | "
            f"Time: {stats['time']:.2f}s"
        )

    print("Training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN using custom deep learning framework"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training dataset root directory"
    )

    args = parser.parse_args()
    main(args)
