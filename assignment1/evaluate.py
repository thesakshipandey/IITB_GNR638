from __future__ import annotations

import argparse

from gnr_dl.backend import G16GetBackend
from gnr_dl.dataloader import DataLoader, G16TrainValTestSplit, ImageFolderDataset
from gnr_dl.loss import CrossEntropyLoss
from gnr_dl.metrics import G16BatchAccuracy
from gnr_dl.model import G16BuildModel, SimpleCNN
from gnr_dl.serialization import G16LoadWeights, G16ReadMetadata
from gnr_dl.utils import G16LoadJsonConfig, G16MergeWithDefaults


DEFAULT_CONFIG = {
    "batch_size": 32,
    "image_size": 32,
    "input_channels": 3,
    "conv_out_channels": 16,
    "conv2_out_channels": 32,
    "use_second_conv": False,
    "fc_hidden_features": 0,
    "model_arch": "custom",
    "kernel_size": 3,
    "conv_stride": 1,
    "conv_padding": 1,
    "pool_size": 2,
    "pool_stride": 2,
    "seed": 42,
    "val_fraction": 0.0,
    "test_fraction": 0.0,
}


def G16ParseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate custom CUDA-ready CNN framework")
    parser.add_argument("--test_dir", required=True, help="Path to parent test dataset folder")
    parser.add_argument("--config", default=None, help="Path to JSON config file (optional, loaded from weights if omitted)")
    parser.add_argument("--weights", required=True, help="Path to saved model weights")
    return parser.parse_args()


def G16PrintComplexity(model: SimpleCNN) -> None:
    stats = model.G16Complexity()
    print(f"Model parameters: {stats['parameters']}")
    print(f"Model MACs/forward/sample: {stats['macs']}")
    print(f"Model FLOPs/forward/sample: {stats['flops']}")


def main() -> None:
    args = G16ParseArgs()

    metadata = G16ReadMetadata(args.weights)

    if args.config is not None:
        user_config = G16LoadJsonConfig(args.config)
    else:
        user_config = metadata.get("config", {})

    config = G16MergeWithDefaults(user_config, DEFAULT_CONFIG)
    class_to_idx = metadata.get("class_to_idx")
    if class_to_idx is not None and not isinstance(class_to_idx, dict):
        raise ValueError("Invalid class_to_idx metadata in weights file")

    dataset = ImageFolderDataset(
        root_dir=args.test_dir,
        image_size=int(config["image_size"]),
        class_to_idx=class_to_idx,
        preload=True,
    )

    print(f"Dataset loading time (seconds): {dataset.loading_time_seconds:.4f}")
    print(f"Total samples: {len(dataset)}")

    val_fraction = float(config.get("val_fraction", 0.0))
    test_fraction = float(config.get("test_fraction", 0.0))
    eval_dataset = dataset
    if test_fraction > 0.0:
        _, _, test_set = G16TrainValTestSplit(
            dataset=dataset,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=int(config.get("seed", 42)),
        )
        eval_dataset = test_set
        print(
            "Split evaluation enabled | "
            f"Seed {int(config.get('seed', 42))} | "
            f"Val fraction {val_fraction:.2f} | Test fraction {test_fraction:.2f}"
        )
    print(f"Evaluation samples: {len(eval_dataset)}")

    requested_arch = str(config.get("model_arch", "custom"))
    model, resolved_arch = G16BuildModel(
        num_classes=dataset.num_classes,
        input_channels=int(config["input_channels"]),
        input_size=int(config["image_size"]),
        model_arch=requested_arch,
        conv_out_channels=int(config["conv_out_channels"]),
        conv2_out_channels=int(config["conv2_out_channels"]),
        use_second_conv=bool(config["use_second_conv"]),
        fc_hidden_features=int(config["fc_hidden_features"]),
        kernel_size=int(config["kernel_size"]),
        conv_stride=int(config["conv_stride"]),
        conv_padding=int(config["conv_padding"]),
        pool_size=int(config["pool_size"]),
        pool_stride=int(config["pool_stride"]),
    )
    print(f"Model architecture: requested={requested_arch} | resolved={resolved_arch}")

    backend = G16GetBackend()
    print(f"Backend mode: {'CUDA' if backend.G16UsingCuda else 'Python fallback'}")
    G16PrintComplexity(model)

    loaded_metadata = G16LoadWeights(model, args.weights)
    if loaded_metadata.get("class_to_idx"):
        print("Loaded class mapping from saved weights")

    criterion = CrossEntropyLoss()
    dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        logits = model.G16Forward(x_batch)
        loss = criterion.G16Forward(logits, y_batch)
        acc = G16BatchAccuracy(logits, y_batch)

        current_batch_size = len(y_batch)
        total_loss += loss * current_batch_size
        total_acc += acc * current_batch_size
        total_samples += current_batch_size

    eval_loss = total_loss / max(total_samples, 1)
    eval_acc = total_acc / max(total_samples, 1)

    print(f"Evaluation loss: {eval_loss:.4f}")
    print(f"Evaluation accuracy: {eval_acc:.4f}")


if __name__ == "__main__":
    main()
