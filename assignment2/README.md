# GNR638 Assignment 2

This GitHub-ready copy contains the full codebase, generated plots, and result tables for:

- Scenario 4.1: linear probe transfer
- Scenario 4.2: fine-tuning strategy comparison
- Scenario 4.3: few-shot learning analysis
- Scenario 4.4: corruption robustness evaluation
- Scenario 4.5: layer-wise feature probing

## What is included

- Source code under `scenarios/`, `utils/`, and the top-level Python scripts
- All generated plots under `results/`
- Summary CSV/JSON artifacts needed to regenerate consolidated plots and tables

## What is intentionally excluded

- `train_data/` and `train_data.zip`
- PyTorch checkpoints (`*.pt`)
- Report sources, report PDFs, and report-only tables
- Backups, archives, caches, and local LaTeX binaries

This keeps the repository small enough for GitHub while preserving the generated figures and experiment outputs.

## Full reproduction from scratch

1. Install the dataset so that the AID image folders live under `train_data/`.
2. Run the following commands:

```bash
cd ~/Documents/GNR638/IITB_GNR638/assignment2
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 run_all.py
python3 generate_all_plots.py --results_dir results --out_dir results/all_plots
```

This runs all scenarios and regenerates the consolidated plots in `results/all_plots`.

## Notes

- For a clean full rerun, you need the AID dataset in `train_data/`.
