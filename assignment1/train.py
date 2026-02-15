from __future__ import annotations

import argparse
import json
import os
import time

from gnr_dl.backend import G16GetBackend
from gnr_dl.dataloader import DataLoader, G16TrainValTestSplit, ImageFolderDataset
from gnr_dl.loss import CrossEntropyLoss
from gnr_dl.metrics import G16BatchAccuracy
from gnr_dl.model import G16BuildModel, SimpleCNN
from gnr_dl.optimizer import SGD
from gnr_dl.serialization import G16LoadWeights, G16SaveWeights
from gnr_dl.utils import G16LoadJsonConfig, G16MergeWithDefaults, G16SetSeed


DEFAULT_CONFIG = {
    "seed": 42,
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0,
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
    "val_fraction": 0.1,
    "test_fraction": 0.1,
    "shuffle": True,
    "print_every": 20,
}


def G16ParseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom CUDA-ready CNN framework")
    parser.add_argument("--train_dir", required=True, help="Path to parent training dataset folder")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--weights_out",
        default="artifacts/weights.json",
        help="Output path for saved model weights",
    )
    parser.add_argument(
        "--metrics_out",
        default=None,
        help="Optional output path for training history JSON (train/val loss+acc by epoch)",
    )
    parser.add_argument(
        "--best_weights_out",
        default=None,
        help="Optional output path for best-checkpoint weights (highest validation accuracy)",
    )
    return parser.parse_args()


def G16PrintComplexity(model: SimpleCNN) -> None:
    stats = model.G16Complexity()
    print(f"Model parameters: {stats['parameters']}")
    print(f"Model MACs/forward/sample: {stats['macs']}")
    print(f"Model FLOPs/forward/sample: {stats['flops']}")


def G16DefaultMetricsPath(weights_out: str) -> str:
    root, ext = os.path.splitext(weights_out)
    if ext:
        return f"{root}_metrics.json"
    return f"{weights_out}_metrics.json"


def G16DefaultBestWeightsPath(weights_out: str) -> str:
    root, ext = os.path.splitext(weights_out)
    if ext:
        return f"{root}_best{ext}"
    return f"{weights_out}_best.json"


def G16EvaluateSubset(
    model: SimpleCNN,
    criterion: CrossEntropyLoss,
    subset_dataset,
    batch_size: int,
) -> tuple[float, float, int]:
    dataloader = DataLoader(dataset=subset_dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        logits = model.G16Forward(x_batch)
        loss = criterion.G16Forward(logits, y_batch)
        acc = G16BatchAccuracy(logits, y_batch)
        n = len(y_batch)
        total_loss += loss * n
        total_acc += acc * n
        total_samples += n

    eval_loss = total_loss / max(total_samples, 1)
    eval_acc = total_acc / max(total_samples, 1)
    return eval_loss, eval_acc, total_samples


def main() -> None:
    args = G16ParseArgs()

    user_config = G16LoadJsonConfig(args.config)
    config = G16MergeWithDefaults(user_config, DEFAULT_CONFIG)
    G16SetSeed(int(config["seed"]))

    dataset = ImageFolderDataset(
        root_dir=args.train_dir,
        image_size=int(config["image_size"]),
        preload=True,
    )
    print(f"Dataset loading time (seconds): {dataset.loading_time_seconds:.4f}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")

    train_set, val_set, test_set = G16TrainValTestSplit(
        dataset=dataset,
        val_fraction=float(config["val_fraction"]),
        test_fraction=float(config["test_fraction"]),
        seed=int(config["seed"]),
    )
    print(
        f"Train split: {len(train_set)} samples | "
        f"Val split: {len(val_set)} samples | "
        f"Test split: {len(test_set)} samples"
    )

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

    optimizer = SGD(
        parameters=model.G16Parameters(),
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = CrossEntropyLoss()

    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    print_every = int(config["print_every"])
    history = []
    best_val_acc = float("-inf")
    best_epoch = 0

    best_weights_out = args.best_weights_out if args.best_weights_out else G16DefaultBestWeightsPath(
        args.weights_out
    )
    best_output_dir = os.path.dirname(best_weights_out)
    if best_output_dir:
        os.makedirs(best_output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=bool(config["shuffle"]),
        )

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        batch_count = 0

        for x_batch, y_batch in train_loader:
            optimizer.G16ZeroGrad()

            logits = model.G16Forward(x_batch)
            loss = criterion.G16Forward(logits, y_batch)
            grad_logits = criterion.G16Backward()

            model.G16Backward(grad_logits)
            optimizer.G16Step()

            batch_acc = G16BatchAccuracy(logits, y_batch)
            current_batch_size = len(y_batch)

            total_loss += loss * current_batch_size
            total_acc += batch_acc * current_batch_size
            total_samples += current_batch_size
            batch_count += 1

            if print_every > 0 and batch_count % print_every == 0:
                avg_loss_so_far = total_loss / max(total_samples, 1)
                avg_acc_so_far = total_acc / max(total_samples, 1)
                print(
                    f"Epoch {epoch} | Batch {batch_count} | "
                    f"Train Loss {avg_loss_so_far:.4f} | Train Acc {avg_acc_so_far:.4f}"
                )

        epoch_time = time.perf_counter() - epoch_start
        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_acc / max(total_samples, 1)

        val_loss, val_acc, val_samples = G16EvaluateSubset(
            model=model,
            criterion=criterion,
            subset_dataset=val_set,
            batch_size=batch_size,
        )

        print(
            f"Epoch {epoch}/{epochs} completed | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | "
            f"Epoch time {epoch_time:.2f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch_time_seconds": epoch_time,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_metadata = {
                "class_to_idx": dataset.class_to_idx,
                "config": config,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_acc,
            }
            G16SaveWeights(model, best_weights_out, metadata=best_metadata)
            print(
                f"Saved new best model | Epoch {best_epoch} | "
                f"Val Acc {best_val_acc:.4f} | Path: {best_weights_out}"
            )

    output_dir = os.path.dirname(args.weights_out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "class_to_idx": dataset.class_to_idx,
        "config": config,
    }
    G16SaveWeights(model, args.weights_out, metadata=metadata)
    print(f"Saved weights to: {args.weights_out}")
    print(
        f"Best checkpoint | Epoch {best_epoch} | Val Acc {best_val_acc:.4f} | "
        f"Path: {best_weights_out}"
    )

    final_test_loss, final_test_acc, test_samples = G16EvaluateSubset(
        model=model,
        criterion=criterion,
        subset_dataset=test_set,
        batch_size=batch_size,
    )
    print(
        f"Final model test metrics | Samples {test_samples} | "
        f"Loss {final_test_loss:.4f} | Acc {final_test_acc:.4f}"
    )

    best_test_loss = None
    best_test_acc = None
    if best_epoch > 0:
        best_model, _ = G16BuildModel(
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
        G16LoadWeights(best_model, best_weights_out)
        best_test_loss, best_test_acc, _ = G16EvaluateSubset(
            model=best_model,
            criterion=criterion,
            subset_dataset=test_set,
            batch_size=batch_size,
        )
        print(
            f"Best model test metrics | Epoch {best_epoch} | "
            f"Loss {best_test_loss:.4f} | Acc {best_test_acc:.4f}"
        )

    metrics_out = args.metrics_out if args.metrics_out else G16DefaultMetricsPath(args.weights_out)
    metrics_dir = os.path.dirname(metrics_out)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    history_payload = {
        "train_dir": args.train_dir,
        "config": config,
        "split": {
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "test_samples": len(test_set),
            "seed": int(config["seed"]),
            "val_fraction": float(config["val_fraction"]),
            "test_fraction": float(config["test_fraction"]),
        },
        "best_model": {
            "epoch": best_epoch,
            "val_accuracy": best_val_acc,
            "weights_path": best_weights_out,
            "test_loss": best_test_loss,
            "test_accuracy": best_test_acc,
        },
        "final_model": {
            "weights_path": args.weights_out,
            "test_loss": final_test_loss,
            "test_accuracy": final_test_acc,
        },
        "epochs": history,
    }
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(history_payload, handle, indent=2)
    print(f"Saved training history to: {metrics_out}")


if __name__ == "__main__":
    main()
