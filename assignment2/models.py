"""
Model factory and parameter-freezing utilities for Assignment 2.
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn

import config


def create_model(
    name: str,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build an ImageNet-pretrained timm model with a new task head."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def _head_attr(model_name: str) -> str:
    heads = {
        "resnet50": "fc",
        "efficientnet_b0": "classifier",
        "convnext_tiny": "head",
    }
    if model_name not in heads:
        raise ValueError(f"Unsupported model: {model_name}")
    return heads[model_name]


def get_classifier_head(model: nn.Module, model_name: str) -> nn.Module:
    return getattr(model, _head_attr(model_name))


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all non-head parameters and keep the classifier head trainable."""
    head_attr = _head_attr(model_name)
    head_module = get_classifier_head(model, model_name)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head_attr + ".") or name == head_attr

    # Handles nested head modules (e.g., ConvNeXt head.fc)
    for param in head_module.parameters():
        param.requires_grad = True


def unfreeze_full(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _get_module(model: nn.Module, path: str) -> nn.Module:
    return reduce(getattr, path.split("."), model)


def unfreeze_last_block(model: nn.Module, model_name: str) -> None:
    last_block_paths = {
        "resnet50": "layer4",
        "efficientnet_b0": "blocks.6",
        "convnext_tiny": "stages.3",
    }
    path = last_block_paths[model_name]
    for p in _get_module(model, path).parameters():
        p.requires_grad = True


def _candidate_paths(model_name: str) -> List[str]:
    """Deep-to-shallow candidate groups used for selective unfreezing."""
    if model_name == "resnet50":
        return [
            "layer4.2", "layer4.1", "layer4.0",
            "layer3.5", "layer3.4", "layer3.3", "layer3.2", "layer3.1", "layer3.0",
            "layer2.3", "layer2.2", "layer2.1", "layer2.0",
            "layer1.2", "layer1.1", "layer1.0",
            "bn1", "conv1",
        ]
    if model_name == "efficientnet_b0":
        return [
            "conv_head", "bn2", "blocks.6.0",
            "blocks.5.3", "blocks.5.2", "blocks.5.1", "blocks.5.0",
            "blocks.4.2", "blocks.4.1", "blocks.4.0",
            "blocks.3.2", "blocks.3.1", "blocks.3.0",
            "blocks.2.1", "blocks.2.0",
            "blocks.1.1", "blocks.1.0", "blocks.0.0",
            "bn1", "conv_stem",
        ]
    if model_name == "convnext_tiny":
        return [
            "stages.3.blocks.2", "stages.3.blocks.1", "stages.3.blocks.0",
            "stages.3.downsample.1", "stages.3.downsample.0",
            "stages.2.blocks.8", "stages.2.blocks.7", "stages.2.blocks.6",
            "stages.2.blocks.5", "stages.2.blocks.4", "stages.2.blocks.3",
            "stages.2.blocks.2", "stages.2.blocks.1", "stages.2.blocks.0",
            "stages.2.downsample.1", "stages.2.downsample.0",
            "stages.1.blocks.2", "stages.1.blocks.1", "stages.1.blocks.0",
            "stages.1.downsample.1", "stages.1.downsample.0",
            "stages.0.blocks.2", "stages.0.blocks.1", "stages.0.blocks.0",
            "stem.1", "stem.0",
        ]
    raise ValueError(f"Unsupported model: {model_name}")


def _backbone_param_total(model: nn.Module, model_name: str) -> int:
    head_attr = _head_attr(model_name)
    return sum(
        p.numel()
        for n, p in model.named_parameters()
        if not n.startswith(head_attr + ".") and n != head_attr
    )


def _available_candidates(model: nn.Module, model_name: str) -> List[Dict]:
    candidates = []
    for depth_rank, path in enumerate(_candidate_paths(model_name)):
        try:
            module = _get_module(model, path)
        except AttributeError:
            continue
        n_params = sum(p.numel() for p in module.parameters())
        if n_params == 0:
            continue
        candidates.append(
            {
                "path": path,
                "module": module,
                "n_params": n_params,
                "depth_rank": depth_rank,  # smaller means deeper
            }
        )
    return candidates


def _score_candidates_with_gradients(
    model: nn.Module,
    candidates: List[Dict],
    calibration_loader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    """Estimate per-group adaptation pressure using gradient norms."""
    grad_sums = {c["path"]: 0.0 for c in candidates}
    used_batches = 0
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (images, labels) in enumerate(calibration_loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        used_batches += 1

        for cand in candidates:
            norms = [
                p.grad.detach().norm().item()
                for p in cand["module"].parameters()
                if p.grad is not None
            ]
            if norms:
                grad_sums[cand["path"]] += float(sum(norms) / len(norms))

    if used_batches == 0:
        return {k: 0.0 for k in grad_sums}
    return {k: v / used_batches for k, v in grad_sums.items()}


def unfreeze_selective(
    model: nn.Module,
    model_name: str,
    target_fraction: float = config.SELECTIVE_UNFREEZE_FRACTION,
    calibration_loader=None,
    device: Optional[torch.device] = None,
    max_calibration_batches: int = 0,
) -> Dict:
    """
    Unfreeze a subset of backbone groups under a strict parameter budget.

    If a calibration loader is provided, candidate groups are ranked using
    gradient-based scores from a few mini-batches; otherwise, deepest-first
    ranking is used.
    """
    total_backbone = _backbone_param_total(model, model_name)
    budget = int(target_fraction * total_backbone)
    candidates = _available_candidates(model, model_name)
    if not candidates:
        raise RuntimeError(f"No selective candidates found for model {model_name}")

    use_gradient_scoring = (
        calibration_loader is not None
        and device is not None
        and max_calibration_batches > 0
    )
    method = "depth_greedy"
    raw_scores: Dict[str, float] = {c["path"]: 0.0 for c in candidates}

    if use_gradient_scoring:
        method = "gradient_budgeted"
        # Temporarily unfreeze all candidate groups to measure gradients.
        for cand in candidates:
            for p in cand["module"].parameters():
                p.requires_grad = True
        raw_scores = _score_candidates_with_gradients(
            model=model,
            candidates=candidates,
            calibration_loader=calibration_loader,
            device=device,
            max_batches=max_calibration_batches,
        )

    # Reset to strict frozen-backbone baseline before final selection.
    freeze_backbone(model, model_name)

    max_raw = max(raw_scores.values()) if raw_scores else 0.0
    densities = {
        c["path"]: (raw_scores[c["path"]] / c["n_params"]) if c["n_params"] > 0 else 0.0
        for c in candidates
    }
    max_density = max(densities.values()) if densities else 0.0
    depth_norm_den = max(len(candidates) - 1, 1)

    candidate_rows = []
    for cand in candidates:
        path = cand["path"]
        raw = raw_scores[path]
        density = densities[path]
        depth_bonus = 1.0 - (cand["depth_rank"] / depth_norm_den)

        if method == "gradient_budgeted" and max_raw > 0.0:
            raw_norm = raw / max_raw
            density_norm = density / max_density if max_density > 0 else 0.0
            score = 0.65 * raw_norm + 0.25 * density_norm + 0.10 * depth_bonus
        else:
            score = depth_bonus

        candidate_rows.append(
            {
                "path": path,
                "n_params": cand["n_params"],
                "depth_rank": cand["depth_rank"],
                "raw_grad_score": raw,
                "score_density": density,
                "composite_score": score,
            }
        )

    ranked = sorted(
        candidate_rows,
        key=lambda r: (r["composite_score"], -r["depth_rank"], r["raw_grad_score"]),
        reverse=True,
    )

    selected_paths = []
    selected_params = 0
    for row in ranked:
        if selected_params + row["n_params"] <= budget:
            selected_paths.append(row["path"])
            selected_params += row["n_params"]

    # Ensure at least one group is selected.
    if not selected_paths:
        feasible = [r for r in ranked if r["n_params"] <= budget]
        fallback = feasible[0] if feasible else min(ranked, key=lambda r: r["n_params"])
        selected_paths = [fallback["path"]]
        selected_params = fallback["n_params"]

    for path in selected_paths:
        for p in _get_module(model, path).parameters():
            p.requires_grad = True

    selected_set = set(selected_paths)
    for row in candidate_rows:
        row["selected"] = row["path"] in selected_set

    actual_fraction = selected_params / total_backbone if total_backbone > 0 else 0.0
    print(f"[Selective Unfreeze] Model: {model_name}")
    print(f"  Method: {method}")
    print(f"  Unfrozen groups: {selected_paths}")
    print(
        f"  Params unfrozen: {selected_params:,} / {total_backbone:,} "
        f"= {actual_fraction*100:.1f}% (budget {budget:,})"
    )

    return {
        "model_name": model_name,
        "method": method,
        "target_fraction": target_fraction,
        "actual_fraction": actual_fraction,
        "total_backbone_params": total_backbone,
        "budget_params": budget,
        "selected_params": selected_params,
        "selected_paths": selected_paths,
        "candidates": sorted(candidate_rows, key=lambda r: r["depth_rank"]),
    }


def get_param_counts(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def trainable_fraction(model: nn.Module, model_name: str) -> float:
    """Fraction of trainable backbone params (head excluded)."""
    head_params = sum(p.numel() for p in get_classifier_head(model, model_name).parameters())
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_total = total - head_params
    backbone_trainable = trainable - head_params
    return backbone_trainable / backbone_total if backbone_total > 0 else 0.0


def get_trainable_param_groups(
    model: nn.Module,
    model_name: str,
) -> Dict[str, List[nn.Parameter]]:
    """
    Return current trainable params split into head/backbone groups.
    """
    head_params = [p for p in get_classifier_head(model, model_name).parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_ids
    ]
    return {"head": head_params, "backbone": backbone_params}
