from __future__ import annotations

import json
from typing import Any, Dict, List


def G16TrainableLayers(model) -> List[Any]:
    layers = []
    for layer in model.layers:
        if hasattr(layer, "weight") and hasattr(layer, "bias"):
            layers.append(layer)
    return layers


def G16SaveWeights(model, output_path: str, metadata: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "metadata": metadata or {},
        "layers": [],
    }

    for layer in G16TrainableLayers(model):
        payload["layers"].append(
            {
                "type": layer.__class__.__name__,
                "weight_shape": list(layer.weight.shape),
                "weight": layer.weight.data,
                "bias_shape": list(layer.bias.shape),
                "bias": layer.bias.data,
            }
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def G16LoadWeights(model, weights_path: str) -> Dict[str, Any]:
    with open(weights_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    saved_layers = payload["layers"]
    model_layers = G16TrainableLayers(model)

    if len(saved_layers) != len(model_layers):
        raise ValueError(
            "Model/weights architecture mismatch: "
            f"model has {len(model_layers)} trainable layers but weights contain {len(saved_layers)}. "
            "Use the same config (especially model_arch and channel sizes) that was used during training."
        )

    for idx, layer in enumerate(model_layers):
        saved = saved_layers[idx]

        if len(saved["weight"]) != len(layer.weight.data):
            raise ValueError(
                "Model/weights architecture mismatch at layer "
                f"{idx}: expected weight size {len(layer.weight.data)}, "
                f"got {len(saved['weight'])}. "
                "Use matching model_arch/conv settings for train and eval."
            )
        if len(saved["bias"]) != len(layer.bias.data):
            raise ValueError(
                "Model/weights architecture mismatch at layer "
                f"{idx}: expected bias size {len(layer.bias.data)}, "
                f"got {len(saved['bias'])}. "
                "Use matching model_arch/conv settings for train and eval."
            )

        layer.weight.data = [float(v) for v in saved["weight"]]
        layer.bias.data = [float(v) for v in saved["bias"]]

    return payload.get("metadata", {})


def G16ReadMetadata(weights_path: str) -> Dict[str, Any]:
    with open(weights_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("metadata", {})


# Backward-compatible aliases
save_weights = G16SaveWeights
load_weights = G16LoadWeights
read_metadata = G16ReadMetadata
