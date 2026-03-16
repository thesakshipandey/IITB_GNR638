#!/usr/bin/env python3
"""Hyperparameter tuning driver for Assignment 2.

This script evaluates candidate global training settings on a proxy objective
that prioritizes transfer + few-shot + robustness:
  - Scenario 4.1 (linear probe transfer)
  - Scenario 4.3 (few-shot analysis)
  - Scenario 4.4 (corruption robustness)

Each candidate runs all three models under fixed seed for fair comparison.
Results are stored under results/tuning_runs/<candidate_name>/.
A ranked summary is saved to results/tuning_summary.csv and
results/best_tuned_config.json.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch

import config
import train as trn
from scenarios import s41_linear_probe, s43_fewshot, s44_corruption


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = Path(config.RESULTS_DIR)
TUNING_ROOT = DEFAULT_RESULTS_DIR / "tuning_runs"


@dataclass
class Candidate:
    name: str
    overrides: Dict[str, Any]
    note: str


CANDIDATES: List[Candidate] = [
    Candidate(
        name="baseline",
        overrides={
            "LR_PROBE": 1e-3,
            "LABEL_SMOOTHING": 0.05,
            "S43_STRONG_AUG": False,
            "S43_WEIGHT_DECAY": 1e-4,
            "S43_LOW_DATA_LR": 1e-3,
            "S43_EARLY_STOP_PATIENCE": 0,
        },
        note="Current stable baseline",
    ),
    Candidate(
        name="mild_lowdata_reg",
        overrides={
            "LR_PROBE": 1.1e-3,
            "LABEL_SMOOTHING": 0.08,
            "S43_STRONG_AUG": True,
            "S43_WEIGHT_DECAY": 3e-4,
            "S43_LOW_DATA_LR": 8e-4,
            "S43_EARLY_STOP_PATIENCE": 5,
        },
        note="Mild regularization for few-shot + slight probe LR increase",
    ),
    Candidate(
        name="stronger_lowdata_reg",
        overrides={
            "LR_PROBE": 1e-3,
            "LABEL_SMOOTHING": 0.10,
            "S43_STRONG_AUG": True,
            "S43_WEIGHT_DECAY": 5e-4,
            "S43_LOW_DATA_LR": 7e-4,
            "S43_EARLY_STOP_PATIENCE": 5,
        },
        note="Stronger regularization for low-data stability",
    ),
]


def _snapshot_config(keys: List[str]) -> Dict[str, Any]:
    return {k: copy.deepcopy(getattr(config, k)) for k in keys}


def _apply_overrides(overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        setattr(config, key, value)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    return pd.read_csv(path)


def _compute_objective(run_dir: Path) -> Dict[str, float]:
    s41 = _safe_read_csv(run_dir / "s41" / "summary_table.csv")
    s43 = _safe_read_csv(run_dir / "s43" / "fewshot_table.csv")
    s44 = _safe_read_csv(run_dir / "s44" / "robustness_summary.csv")

    s41_mean_best = float(s41["Best Val Acc (%)"].mean())

    s43_20 = s43[s43["Data Fraction"].astype(str) == "20%"]
    s43_5 = s43[s43["Data Fraction"].astype(str) == "5%"]
    s43_mean_20 = float(s43_20["Best Val Acc (%)"].mean())
    s43_mean_5 = float(s43_5["Best Val Acc (%)"].mean())
    s43_gap_5 = float(s43_5["Train-Val Gap (%)"].mean())

    rr_mean = float(s44["Relative Robustness"].mean())
    rr_mean_pct = rr_mean * 100.0

    # Weighted score focused on assignment pain points (few-shot + transfer)
    score = (
        0.35 * s41_mean_best
        + 0.40 * s43_mean_5
        + 0.15 * s43_mean_20
        + 0.10 * rr_mean_pct
        - 0.05 * s43_gap_5
    )

    return {
        "s41_mean_best_val": s41_mean_best,
        "s43_mean_best_val_20": s43_mean_20,
        "s43_mean_best_val_5": s43_mean_5,
        "s43_mean_gap_5": s43_gap_5,
        "s44_mean_relative_robustness": rr_mean,
        "objective_score": score,
    }


def _run_candidate(candidate: Candidate, device: torch.device) -> Dict[str, Any]:
    run_dir = TUNING_ROOT / candidate.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Candidate-specific result root
    config.RESULTS_DIR = str(run_dir)
    _apply_overrides(candidate.overrides)

    print("\n" + "=" * 70)
    print(f"TUNING CANDIDATE: {candidate.name}")
    print(f"Results dir: {run_dir}")
    print(f"Note: {candidate.note}")
    print("Overrides:")
    for k, v in candidate.overrides.items():
        print(f"  - {k} = {v}")
    print("=" * 70)

    # Fixed seed for fair candidate comparison
    trn.set_seed(config.SEED)

    scenario_times = {}

    t0 = time.time()
    s41_linear_probe.run(model_names=config.MODELS, device=device)
    scenario_times["41"] = time.time() - t0

    t0 = time.time()
    s43_fewshot.run(model_names=config.MODELS, device=device)
    scenario_times["43"] = time.time() - t0

    t0 = time.time()
    s44_corruption.run(model_names=config.MODELS, device=device)
    scenario_times["44"] = time.time() - t0

    metrics = _compute_objective(run_dir)
    total_time = sum(scenario_times.values())

    result = {
        "candidate": candidate.name,
        "note": candidate.note,
        "results_dir": str(run_dir),
        **candidate.overrides,
        **{f"time_s{sid}_sec": round(sec, 1) for sid, sec in scenario_times.items()},
        "time_total_sec": round(total_time, 1),
        "time_total_hr": round(total_time / 3600.0, 3),
        **{k: round(v, 4) for k, v in metrics.items()},
    }

    with open(run_dir / "candidate_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(TUNING_ROOT, exist_ok=True)

    keys_to_restore = [
        "RESULTS_DIR",
        "LR_PROBE",
        "LABEL_SMOOTHING",
        "S43_STRONG_AUG",
        "S43_WEIGHT_DECAY",
        "S43_LOW_DATA_LR",
        "S43_EARLY_STOP_PATIENCE",
    ]
    original = _snapshot_config(keys_to_restore)

    rows: List[Dict[str, Any]] = []

    try:
        for cand in CANDIDATES:
            row = _run_candidate(cand, device=device)
            rows.append(row)
    finally:
        # Restore global config no matter what
        for key, value in original.items():
            setattr(config, key, value)

    df = pd.DataFrame(rows)
    df = df.sort_values(by="objective_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    summary_csv = DEFAULT_RESULTS_DIR / "tuning_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)

    best = df.iloc[0].to_dict()
    best_config = {
        "selected_candidate": best["candidate"],
        "objective_score": float(best["objective_score"]),
        "overrides": {
            "LR_PROBE": float(best["LR_PROBE"]),
            "LABEL_SMOOTHING": float(best["LABEL_SMOOTHING"]),
            "S43_STRONG_AUG": bool(best["S43_STRONG_AUG"]),
            "S43_WEIGHT_DECAY": float(best["S43_WEIGHT_DECAY"]),
            "S43_LOW_DATA_LR": float(best["S43_LOW_DATA_LR"]),
            "S43_EARLY_STOP_PATIENCE": int(best["S43_EARLY_STOP_PATIENCE"]),
        },
    }

    best_json = DEFAULT_RESULTS_DIR / "best_tuned_config.json"
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print("\nTuning complete.")
    print(f"Summary: {summary_csv}")
    print(f"Best config: {best_json}")
    print(df[["rank", "candidate", "objective_score", "s41_mean_best_val", "s43_mean_best_val_5", "s44_mean_relative_robustness", "time_total_hr"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
