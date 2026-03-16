"""
Efficiency metrics reporting for GNR638 Assignment 2.

Mirrors the reporting pattern from Assgn1 (complexity_metrics.py):
  - Parameter count
  - MACs (Multiply-Accumulate Operations)
  - FLOPs (≈ 2 × MACs)

Uses the ptflops library (pip install ptflops).
"""

import torch
import torch.nn as nn

import config

try:
    from ptflops import get_model_complexity_info
    _PTFLOPS_AVAILABLE = True
except ImportError:
    _PTFLOPS_AVAILABLE = False
    print("[efficiency] WARNING: ptflops not installed. Run: pip install ptflops")


def report_efficiency(model: nn.Module, model_name: str,
                      input_size: tuple = (3, config.IMAGE_SIZE, config.IMAGE_SIZE)) -> dict:
    """
    Compute and print parameters, MACs, FLOPs for a model.

    Args:
        model:       the nn.Module (should be on CPU for ptflops)
        model_name:  display name string
        input_size:  (C, H, W) without batch dimension

    Returns:
        dict with keys 'params', 'macs', 'flops'
    """
    # Parameter count (always available)
    total_params = sum(p.numel() for p in model.parameters())

    macs_val = None
    flops_val = None
    first_param = next(model.parameters(), None)
    original_device = first_param.device if first_param is not None else torch.device("cpu")

    if _PTFLOPS_AVAILABLE:
        try:
            # ptflops expects model on CPU
            cpu_model = model.cpu()
            macs_raw, params_raw = get_model_complexity_info(
                cpu_model,
                input_size,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
            macs_val  = macs_raw
            flops_val = 2 * macs_raw
        except Exception as e:
            print(f"[efficiency] ptflops error for {model_name}: {e}")
        finally:
            # Restore the caller's original device because ptflops moves the module.
            if original_device.type != "cpu":
                model.to(original_device)
    else:
        print(f"[efficiency] Skipping MACs/FLOPs for {model_name} (ptflops not available)")

    # Format for display (same style as Assgn1 complexity_metrics.py)
    def _fmt_num(n, unit="M"):
        if n is None:
            return "N/A"
        if unit == "G":
            return f"{n/1e9:.3f}G"
        return f"{n/1e6:.2f}M"

    print(
        f"[Efficiency] {model_name:20s} | "
        f"Params: {_fmt_num(total_params, 'M'):>10s} | "
        f"MACs: {_fmt_num(macs_val, 'G'):>10s} | "
        f"FLOPs: {_fmt_num(flops_val, 'G'):>10s}"
    )

    return {
        "model":  model_name,
        "params": total_params,
        "macs":   macs_val,
        "flops":  flops_val,
    }
