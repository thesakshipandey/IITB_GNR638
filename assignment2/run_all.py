"""
Master runner for GNR638 Assignment 2.

Runs all 5 experimental scenarios in order for all 3 backbone models.
Tracks wall-clock time per scenario/model and saves a computational budget report.

Usage:
    python run_all.py                          # run all scenarios, all models
    python run_all.py --scenarios 41 42        # run only scenarios 4.1 and 4.2
    python run_all.py --models resnet50        # run only one model
    python run_all.py --smoke                  # smoke test (2 epochs, 1 model)
"""

import argparse
import os
import time
import json
import pandas as pd
import torch

import config
import train as trn

# Scenario modules
from scenarios import s41_linear_probe, s42_finetuning, s43_fewshot, s44_corruption, s45_layer_probing


SCENARIO_MAP = {
    "41": ("4.1 Linear Probe Transfer",    s41_linear_probe.run),
    "42": ("4.2 Fine-Tuning Strategies",   s42_finetuning.run),
    "43": ("4.3 Few-Shot Learning",        s43_fewshot.run),
    "44": ("4.4 Corruption Robustness",    s44_corruption.run),
    "45": ("4.5 Layer-Wise Probing",       s45_layer_probing.run),
}


def parse_args():
    parser = argparse.ArgumentParser(description="GNR638 Assgn2 — Run All Scenarios")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_MAP.keys()),
                        help="Scenario IDs to run (e.g. 41 42 43 44 45)")
    parser.add_argument("--models", nargs="+", default=config.MODELS,
                        help="Model names (e.g. resnet50 efficientnet_b0 convnext_tiny)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 2 epochs, first model only")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"Random seed (default: {config.SEED})")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override config for smoke test
    if args.smoke:
        print("\n[SMOKE TEST MODE] — 2 epochs, first model only")
        config.EPOCHS_FULL = 2
        config.EPOCHS_FEWSHOT = 2
        args.models = [args.models[0]]
        args.scenarios = list(SCENARIO_MAP.keys())

    # Apply seed override
    config.SEED = args.seed
    trn.set_seed(config.SEED)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GNR638 ASSIGNMENT 2 — Transfer Learning & Robustness Analysis")
    print(f"{'='*70}")
    print(f"Device:    {device}")
    print(f"Seed:      {config.SEED}")
    print(f"Models:    {args.models}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Results → {config.RESULTS_DIR}")
    print(f"{'='*70}")

    budget_rows = []
    total_start = time.time()

    for scenario_id in args.scenarios:
        if scenario_id not in SCENARIO_MAP:
            print(f"WARNING: Unknown scenario {scenario_id}, skipping.")
            continue

        scenario_name, run_fn = SCENARIO_MAP[scenario_id]
        print(f"\n{'─'*70}")
        print(f"  Running Scenario {scenario_name}")
        print(f"{'─'*70}")

        t0 = time.time()
        try:
            run_fn(model_names=args.models, device=device)
            status = "OK"
        except Exception as e:
            import traceback
            print(f"\nERROR in scenario {scenario_id}: {e}")
            traceback.print_exc()
            status = f"ERROR: {e}"

        elapsed = time.time() - t0
        hours = elapsed / 3600
        print(f"\n  Scenario {scenario_id} finished in {elapsed:.1f}s ({hours:.2f}h)")

        budget_rows.append({
            "Scenario": scenario_name,
            "Models": ", ".join(args.models),
            "Wall Time (s)": round(elapsed, 1),
            "Wall Time (h)": round(hours, 3),
            "Status": status,
        })

    total_elapsed = time.time() - total_start
    budget_rows.append({
        "Scenario": "TOTAL",
        "Models": ", ".join(args.models),
        "Wall Time (s)": round(total_elapsed, 1),
        "Wall Time (h)": round(total_elapsed / 3600, 3),
        "Status": "—",
    })

    # Save computational budget report
    budget_path = os.path.join(config.RESULTS_DIR, "computational_budget.csv")
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(budget_path, index=False)

    print(f"\n{'='*70}")
    print(f"ALL SCENARIOS COMPLETE  — Total time: {total_elapsed/3600:.2f}h")
    print(f"Computational budget saved to: {budget_path}")
    print(f"{'='*70}")
    print(budget_df.to_string(index=False))


if __name__ == "__main__":
    main()
